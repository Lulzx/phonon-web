/* tslint:disable */
/* eslint-disable */

/**
 * Holds the prediction / joint weights (int8 codes + per-row scales).
 */
export class Decoder {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * enc: projected encoder output [T*640].  Returns [n, tok_0..tok_n-1, frame_0.., dur_0..].
     */
    greedy(enc: Float32Array, t_len: number): Uint32Array;
    /**
     * emb f32 [8193*640]; matrices as (int8 codes, per-row f32 scales); b0/b1 = bias_ih + bias_hh
     */
    constructor(emb: Float32Array, wih0: Int8Array, wih0s: Float32Array, whh0: Int8Array, whh0s: Float32Array, b0: Float32Array, wih1: Int8Array, wih1s: Float32Array, whh1: Int8Array, whh1s: Float32Array, b1: Float32Array, pw: Int8Array, pws: Float32Array, pb: Float32Array, jw: Int8Array, jws: Float32Array, jb: Float32Array);
}

/**
 * Decode an entropy-coded integer table (symbols 0..2^bits, offset -2^(bits-1)) and dequantize with per-row (or
 * per-group) f32 scales -> f32 values.
 */
export function decode_dense(blob: Uint8Array, n: number, bits: number, scales: Float32Array, group: number): Float32Array;

/**
 * Decode an entropy-coded integer table to its signed integer codes (|q| <= 127).
 */
export function decode_dense_q(blob: Uint8Array, n: number, bits: number): Int8Array;

/**
 * Decode one entropy-coded five-value matrix into GPU layout.  Returns the u32 words: plane A ([O][I/16], 2-bit
 * codes 0=zero 1=+ 2=-) followed by plane B ([O][I/32], 1 = |w| is hi).
 */
export function decode_five(blob: Uint8Array, rows: number, cols: number): Uint32Array;

export function features(pcm: Float32Array): Float32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_decoder_free: (a: number, b: number) => void;
    readonly decode_dense: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
    readonly decode_dense_q: (a: number, b: number, c: number, d: number) => [number, number];
    readonly decode_five: (a: number, b: number, c: number, d: number) => [number, number];
    readonly decoder_greedy: (a: number, b: number, c: number, d: number) => [number, number];
    readonly decoder_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number, t: number, u: number, v: number, w: number, x: number, y: number, z: number, a1: number, b1: number, c1: number, d1: number, e1: number, f1: number, g1: number, h1: number) => number;
    readonly features: (a: number, b: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
