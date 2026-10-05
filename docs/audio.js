// Audio loading: any format the browser can decode -> 16 kHz mono Float32Array.
export function parseWav(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, false) !== 0x52494646 || dv.getUint32(8, false) !== 0x57415645) return null;
  let p = 12, fmt = null;
  while (p + 8 <= buf.byteLength) {
    const id = dv.getUint32(p, false), sz = dv.getUint32(p + 4, true);
    if (id === 0x666d7420) fmt = { tag: dv.getUint16(p + 8, true), ch: dv.getUint16(p + 10, true),
      sr: dv.getUint32(p + 12, true), bits: dv.getUint16(p + 22, true) };
    if (id === 0x64617461 && fmt) {
      if (fmt.sr !== 16000 || !(fmt.tag === 1 && fmt.bits === 16) && !(fmt.tag === 3 && fmt.bits === 32)) return null;
      const n = Math.floor(sz / (fmt.bits / 8) / fmt.ch), out = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < fmt.ch; c++) {
          const o = p + 8 + (i * fmt.ch + c) * (fmt.bits / 8);
          s += fmt.bits === 16 ? dv.getInt16(o, true) / 32768 : dv.getFloat32(o, true);
        }
        out[i] = s / fmt.ch;
      }
      return out;
    }
    p += 8 + sz + (sz & 1);
  }
  return null;
}

export async function decodeAudio(buf) {
  const direct = parseWav(buf);
  if (direct) return direct;
  const ctx = new (globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext)(1, 16000, 16000);
  const ab = await ctx.decodeAudioData(buf.slice(0));
  const n = Math.ceil(ab.duration * 16000);
  const off = new OfflineAudioContext(1, n, 16000);
  const src = off.createBufferSource();
  src.buffer = ab;
  src.connect(off.destination);
  src.start();
  const r = await off.startRendering();
  return r.getChannelData(0).slice();
}
