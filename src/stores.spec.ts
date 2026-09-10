import { MemoryStore, RedisStore, SqlStore } from './stores.js';
import type { SessionState } from './types.js';

function state(id = 's1'): SessionState {
  return { version: 1, id, recent: [], archive: [], summary: 'sum', facts: [], seq: 1, compactions: 0, createdAt: 1, updatedAt: 2 };
}

describe('MemoryStore', () => {
  it('round-trips and hands out copies', () => {
    const store = new MemoryStore();
    expect(store.load('s1')).toBeNull();
    const s = state();
    store.save(s);
    const loaded = store.load('s1')!;
    expect(loaded).toEqual(s);
    loaded.summary = 'changed';
    expect(store.load('s1')!.summary).toBe('sum');
    expect(store.ids()).toEqual(['s1']);
    store.delete('s1');
    expect(store.load('s1')).toBeNull();
  });
});

describe('SqlStore', () => {
  it('uses $n placeholders and ON CONFLICT for postgres and reads {rows}', async () => {
    const calls: { sql: string; params: unknown[] }[] = [];
    let row: unknown = undefined;
    const store = new SqlStore({
      query: async (sql, params) => {
        calls.push({ sql, params });
        return { rows: row === undefined ? [] : [row] };
      },
    });
    expect(await store.load('s1')).toBeNull();
    await store.save(state());
    row = { state: JSON.stringify(state()) };
    expect(await store.load('s1')).toEqual(state());
    row = { state: state() };
    expect(await store.load('s1')).toEqual(state());
    await store.delete('s1');
    expect(calls.map((c) => c.sql)).toEqual([
      'SELECT state FROM memoryline_sessions WHERE id = $1',
      'INSERT INTO memoryline_sessions (id, state, updated_at) VALUES ($1, $2, $3) ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state, updated_at = EXCLUDED.updated_at',
      'SELECT state FROM memoryline_sessions WHERE id = $1',
      'SELECT state FROM memoryline_sessions WHERE id = $1',
      'DELETE FROM memoryline_sessions WHERE id = $1',
    ]);
    expect(calls[1]!.params).toEqual(['s1', JSON.stringify(state()), 2]);
  });

  it('uses ? placeholders, ON DUPLICATE KEY for mysql, a custom table, and reads plain arrays', async () => {
    const calls: string[] = [];
    const store = new SqlStore({
      dialect: 'mysql',
      table: 'chat_memory',
      query: async (sql) => {
        calls.push(sql);
        return [{ state: JSON.stringify(state()) }];
      },
    });
    expect(await store.load('s1')).toEqual(state());
    await store.save(state());
    expect(calls[0]).toBe('SELECT state FROM chat_memory WHERE id = ?');
    expect(calls[1]).toBe(
      'INSERT INTO chat_memory (id, state, updated_at) VALUES (?, ?, ?) ON DUPLICATE KEY UPDATE state = VALUES(state), updated_at = VALUES(updated_at)',
    );
  });

  it('uses ? placeholders with ON CONFLICT for sqlite and ships a DDL', async () => {
    const calls: string[] = [];
    const store = new SqlStore({ dialect: 'sqlite', query: async (sql) => (calls.push(sql), []) });
    await store.save(state());
    expect(calls[0]).toContain('VALUES (?, ?, ?) ON CONFLICT (id)');
    expect(SqlStore.ddl()).toBe('CREATE TABLE IF NOT EXISTS memoryline_sessions (id VARCHAR(255) PRIMARY KEY, state TEXT NOT NULL, updated_at BIGINT NOT NULL)');
    expect(SqlStore.ddl('t')).toContain('EXISTS t (');
  });
});

describe('RedisStore', () => {
  function fakeClient() {
    const data = new Map<string, string>();
    const expires: [string, number][] = [];
    return {
      data,
      expires,
      get: async (k: string) => data.get(k) ?? null,
      set: async (k: string, v: string) => void data.set(k, v),
      del: async (k: string) => void data.delete(k),
      expire: async (k: string, s: number) => void expires.push([k, s]),
    };
  }

  it('prefixes keys and refreshes the ttl on save', async () => {
    const client = fakeClient();
    const store = new RedisStore(client, { ttlSeconds: 60 });
    expect(await store.load('s1')).toBeNull();
    await store.save(state());
    expect([...client.data.keys()]).toEqual(['memoryline:s1']);
    expect(client.expires).toEqual([['memoryline:s1', 60]]);
    expect(await store.load('s1')).toEqual(state());
    await store.delete('s1');
    expect(client.data.size).toBe(0);
  });

  it('accepts a custom prefix and works without expire when no ttl is set', async () => {
    const { get, set, del } = fakeClient();
    const store = new RedisStore({ get, set, del }, { prefix: 'm:' });
    await store.save(state());
    expect(await store.load('s1')).toEqual(state());
  });

  it('refuses a ttl when the client has no expire', () => {
    const { get, set, del } = fakeClient();
    expect(() => new RedisStore({ get, set, del }, { ttlSeconds: 5 })).toThrow(/expire/);
  });
});
