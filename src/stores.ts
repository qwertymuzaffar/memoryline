import type { SessionState, Store } from './types.js';

/** In-process store. Each load returns a fresh copy, so callers never alias stored state. */
export class MemoryStore implements Store {
  private readonly rows = new Map<string, string>();

  load(id: string): SessionState | null {
    const json = this.rows.get(id);
    return json ? (JSON.parse(json) as SessionState) : null;
  }

  save(state: SessionState): void {
    this.rows.set(state.id, JSON.stringify(state));
  }

  delete(id: string): void {
    this.rows.delete(id);
  }

  ids(): string[] {
    return [...this.rows.keys()];
  }
}

export type SqlDialect = 'postgres' | 'sqlite' | 'mysql';

export interface SqlQuery {
  (sql: string, params: unknown[]): Promise<{ rows: unknown[] } | unknown[]>;
}

export interface SqlStoreOptions {
  /** Runs one statement. Return `{ rows }` (node-postgres) or a plain array (mysql2, better-sqlite3 wrappers). */
  query: SqlQuery;
  /** Default `memoryline_sessions`. */
  table?: string;
  /** Picks placeholders and the upsert syntax. Default `postgres`. */
  dialect?: SqlDialect;
}

/**
 * One row per session with the state as a JSON string. Bring your own query function; no driver is
 * imported. Create the table with `SqlStore.ddl()`.
 */
export class SqlStore implements Store {
  private readonly query: SqlQuery;
  private readonly table: string;
  private readonly dialect: SqlDialect;

  constructor(options: SqlStoreOptions) {
    this.query = options.query;
    this.table = options.table ?? 'memoryline_sessions';
    this.dialect = options.dialect ?? 'postgres';
  }

  static ddl(table = 'memoryline_sessions'): string {
    return `CREATE TABLE IF NOT EXISTS ${table} (id VARCHAR(255) PRIMARY KEY, state TEXT NOT NULL, updated_at BIGINT NOT NULL)`;
  }

  private p(n: number): string {
    return this.dialect === 'postgres' ? `$${n}` : '?';
  }

  async load(id: string): Promise<SessionState | null> {
    const result = await this.query(`SELECT state FROM ${this.table} WHERE id = ${this.p(1)}`, [id]);
    const rows = Array.isArray(result) ? result : result.rows;
    const row = rows[0] as { state?: unknown } | undefined;
    if (!row || row.state == null) return null;
    return (typeof row.state === 'string' ? JSON.parse(row.state) : row.state) as SessionState;
  }

  async save(state: SessionState): Promise<void> {
    const json = JSON.stringify(state);
    const upsert =
      this.dialect === 'mysql'
        ? 'ON DUPLICATE KEY UPDATE state = VALUES(state), updated_at = VALUES(updated_at)'
        : 'ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state, updated_at = EXCLUDED.updated_at';
    await this.query(
      `INSERT INTO ${this.table} (id, state, updated_at) VALUES (${this.p(1)}, ${this.p(2)}, ${this.p(3)}) ${upsert}`,
      [state.id, json, state.updatedAt],
    );
  }

  async delete(id: string): Promise<void> {
    await this.query(`DELETE FROM ${this.table} WHERE id = ${this.p(1)}`, [id]);
  }
}

/** The subset of node-redis and ioredis that RedisStore uses. */
export interface RedisLike {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<unknown>;
  del(key: string): Promise<unknown>;
  expire?(key: string, seconds: number): Promise<unknown>;
}

export interface RedisStoreOptions {
  /** Default `memoryline:`. */
  prefix?: string;
  /** Sliding expiry refreshed on every save. Needs `expire` on the client. */
  ttlSeconds?: number;
}

export class RedisStore implements Store {
  private readonly prefix: string;
  private readonly ttl: number | undefined;

  constructor(
    private readonly client: RedisLike,
    options: RedisStoreOptions = {},
  ) {
    this.prefix = options.prefix ?? 'memoryline:';
    this.ttl = options.ttlSeconds;
    if (this.ttl !== undefined && typeof client.expire !== 'function') {
      throw new TypeError('memoryline: ttlSeconds needs a client with expire(key, seconds)');
    }
  }

  async load(id: string): Promise<SessionState | null> {
    const json = await this.client.get(this.prefix + id);
    return json ? (JSON.parse(json) as SessionState) : null;
  }

  async save(state: SessionState): Promise<void> {
    const key = this.prefix + state.id;
    await this.client.set(key, JSON.stringify(state));
    if (this.ttl !== undefined) await this.client.expire!(key, this.ttl);
  }

  async delete(id: string): Promise<void> {
    await this.client.del(this.prefix + id);
  }
}
