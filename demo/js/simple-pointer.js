import PosenetPointer from "../../src/posenet-pointer";

export default {
  /**
   * Simple example that positions a cursor
   */
  start() {
    this.pointer = new PosenetPointer();
    this.pointer.start();
    this.pointer.use("pointer", poses => {
      console.log(poses[0]);
      window.$pointer.style.top = `${poses[0].pointedAt.y}px`;
      window.$pointer.style.left = `${poses[0].pointedAt.x}px`;
    });
  }
};
