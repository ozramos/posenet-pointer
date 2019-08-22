class PosenetPointer {
  constructor(opts = {}) {
    // Whether this device is supported
    this.isSupported = this.isWebGLSupported();
    // Whether we're running inference or not
    this._isRunning = false;
    // Plugins added with .use()
    this.plugins = {};
    // Stack of poses for smoothing
    this.poseStack = [];
    // Sanitize options and set defaults
    this.configure(opts);
  }

  /**
   * Starts running inference with PoseNet
   * @param {Function} cb A callback to call once PoseNet is ready
   */
  async start(cb) {
    await this.setupFeed();
    await this.initPosenet();
    this._isTracking = true;
    await this.trackPosesLoop(this);
    cb && cb();
  }

  /**
   * Recursive method for tracking poses on each animationFrame:
   * - This method is recursive, once called it continues until after
   *    pointer.stop() is called or until this._isTracking is false
   *
   * @param {PosenetPointer} context The this context, since we're in the
   *    constructor scope now
   */
  trackPosesLoop(context) {
    context.posenet && context.trackPoses();
    if (context.poses) {
      context.runCalculations();
      context.emitEvents();
      context.runPlugins();
    }

    context._isTracking &&
      requestAnimationFrame(() => this.trackPosesLoop(context));
  }

  /**
   * Either assigns passed poses or estimates new poses
   * - Automatically adjusts algorithm to match "single" or "multiple mode"
   * - If debug is on, displays the points and skeletons overlays on the webcam
   */
  async trackPoses() {
    let poses = [];

    // Get single pose
    if (this.options.posenet.maxUsers === 1) {
      let pose = await this.posenet.estimateSinglePose(
        this.video,
        this.options.posenet.imageScaleFactor,
        false,
        this.options.posenet.outputStride
      );
      poses = [pose];
      // Get multiple poses
    } else {
      poses = await this.posenet.estimateMultiplePoses(
        this.video,
        this.options.posenet.imageScaleFactor,
        false,
        this.options.posenet.outputStride,
        this.options.posenet.maxUsers,
        this.options.posenet.scoreThreshold,
        this.options.posenet.nmsRadius
      );
    }

    // Publicly set poses
    this.poses = poses;

    // Only draw when debug is on
    this.options.debug && poses && this.debugPoses();
  }

  /**
   * Runs all plugins
   */
  runPlugins() {
    Object.keys(this.plugins).forEach(key => {
      this.plugins[key].onFrame.call(this, this.poses);
    });
  }

  /**
   * @TODO Emits events
   * - Emits onSeeClarkePoseUpdates with (this.poses, seeclarke)
   */
  emitEvents() {
    window.dispatchEvent(
      new CustomEvent("posenetPointerUpdated", {
        detail: {
          context: this
        }
      })
    );
  }

  /**
   * Adds a plugin with `name` that calles `cb` on every frame
   * - Repeated calls to same `name` overwrites previous
   */
  use(name, cb) {
    this.plugins[name] = {
      onFrame: cb
    };
  }
}

require("./methods/setup")(PosenetPointer);
require("./methods/debug")(PosenetPointer);
require("./methods/calculations")(PosenetPointer);
module.exports = PosenetPointer;
