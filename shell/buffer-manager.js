export class BufferManager {
  constructor(device) {
    this.device = device;
    this.buffers = new Map();
  }

  create(name, size, usage, mappedAtCreation = false) {
    const buffer = this.device.createBuffer({ size, usage, mappedAtCreation });
    this.buffers.set(name, buffer);
    return buffer;
  }

  get(name) {
    const buffer = this.buffers.get(name);
    if (!buffer) throw new Error(`Missing buffer: ${name}`);
    return buffer;
  }

  destroyAll() {
    for (const buffer of this.buffers.values()) {
      buffer.destroy();
    }
    this.buffers.clear();
  }
}
