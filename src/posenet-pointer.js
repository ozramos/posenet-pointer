import * as Posenet from "@tensorflow-models/posenet";
import { merge } from "lodash";

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

  /**
   * Loops through each pose and draws their keypoints/skeletons
   * - Draws skeletons and keypoints
   */
  debugPoses() {
    const context = this.canvas.getContext("2d");

    this.poses.forEach(({ score, keypoints }) => {
      if (score >= this.options.posenet.minPoseConfidence) {
        const adjacentKeypoints = Posenet.getAdjacentKeyPoints(
          keypoints,
          this.options.posenet.minPartConfidence,
          context
        );

        context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawSkeleton(adjacentKeypoints, context);
        this.drawKeypoints(
          keypoints,
          this.options.posenet.minPartConfidence,
          context
        );
      }
    });
  }

  /**
   * Sets up the webcam and stream:
   * - This creates its own video/canvas elements, allowing you to have
   *    multiple instances going (for example, to use front/back cameras
   *    simultaneously)
   * - Recreates the video feed to reassign srcObject once it's been stopped
   */
  async setupFeed() {
    const isMobile = this.isMobile();

    this.canvas.width = this.video.width = 600;
    this.canvas.height = this.video.height = 500;

    // Set window size
    this.cache.window.area = window.outerWidth * window.outerHeight;
    window.addEventListener("resize", () => {
      this.cache.window.area = window.outerWidth * window.outerHeight;
    });

    this.video.srcObject = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user" },
      width: isMobile ? null : this.video.width,
      height: isMobile ? null : this.video.height
    });

    this.video.play();
  }

  /**
   * Init PoseNet
   */
  async initPosenet() {
    if (!this.posenet) {
      this.posenet = await Posenet.load({
        architecture: this.options.posenet.architecture,
        multiplier: this.options.posenet.multiplier,
        outputStride: this.options.posenet.outputStride
      });
    }
  }

  /**
   * Sets defaults to the missing constructor options:
   * - Sets defaults
   * - Creates a default debug container
   * - Creates a default video and canvas
   *
   * @param {Object} opts The options passed into the constructor. Pass null to use all defaults
   */
  configure(opts) {
    // Fallback for default target
    if (!opts.target) {
      opts.target = document.getElementById("posenet-pointer-debug");
      if (!opts.target) {
        opts.target = document.createElement("p");
        opts.target.style.position = "relative";
        document.body.appendChild(opts.target);
      }
    }

    const $video = opts.video || this.createDefaultVideo(opts.target);
    const $canvas = opts.canvas || this.createDefaultCanvas(opts.target);
    this.video = $video;
    this.canvas = $canvas;

    // Setup defaults
    this.options = merge(
      {
        autostart: false,
        canvas: $canvas,
        video: $video,
        debug: false,
        facingMode: "user",
        poseStackSize: 8,
        posenet: {
          architecture: "MobileNetV1",
          multiplier: 0.75,
          maxUsers: 1,
          minPoseConfidence: 0.1,
          minPartConfidence: 0.5,
          outputStride: 16,
          inputResolution: 257,
          nmsRadius: 20,
          scoreThreshold: 0.5
        },
        target: opts.target
      },
      opts
    );

    // Chache
    this.cache = {
      window: {
        // @SEE this.setupFeed
        area: 0
      }
    };
    opts.target.style.display = this.options.debug ? "initial" : "none";
  }

  /**
   * Creates a default (flipped) video and adds it to the DOM:
   * - The video is absolutely positioned within the $wrap
   *
   * @param {HTMLElement} $wrap A container to embed the video into
   *
   * @return {HTMLVideoElement} A hidden video used for inference with PoseNet
   */
  createDefaultVideo($wrap) {
    const $video = document.createElement("video");

    $wrap.classList.add("posenet-pointer-debug");
    $video.setAttribute("playsinline", true);
    $video.style.transform = "scale(-1, 1)";
    $video.style.position = "absolute";
    $wrap.appendChild($video);

    return $video;
  }

  /**
   * Creates a default (flipped) canvas and adds it to the DOM
   * - The canvas is added to the $wrap (along with the video) relatively
   *
   * @param {Element} $wrap The wrapping element to inject the canvas into
   *
   * @return {HTMLCanvasElement} A hidden canvas used for debugging with PoseNet
   */
  createDefaultCanvas($wrap) {
    const $canvas = document.createElement("canvas");
    $canvas.style.transform = "scale(-1, 1)";
    $canvas.style.position = "relative";
    $canvas.style.top = 0;
    $canvas.style.left = 0;

    $wrap.appendChild($canvas);

    return $canvas;
  }

  /**
   * Helpers for checking if we're on mobile
   * - Checks if we're on mobile
   * - Checks if we're on android
   * - Checks if we're on iOS
   */
  isMobile() {
    return this.isAndroid() || this.isIOS();
  }
  isAndroid() {
    return /Android/i.test(navigator.userAgent);
  }
  isIOS() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
  }

  /**
   * Checks if WebGL is supported. Depending on your deployment needs,
   * you can first check if WebGL is supported with this method, and then either
   * display a message or start the tracker.
   *
   * - This will automatically fail if canvas is not supported!
   * - Checks for webgl and experimental-webgl
   *
   * @see https://stackoverflow.com/a/22953053
   *
   * @param {Boolean} forceThrow (Optional) Whether to force throw an error. This is mostly for unit testing to test failures
   * @return {Boolean} Is WebGL supported?
   */
  isWebGLSupported() {
    try {
      let canvas = document.createElement("canvas");
      let isSupported = true;
      if (
        !canvas.getContext("webgl") ||
        !canvas.getContext("experimental-webgl")
      )
        isSupported = false;
      canvas.remove();

      return !!isSupported;
    } catch (e) {
      console.error(e);
      return false;
    }
  }

  /**
   * Draw each tracked keypoint
   * - Draws keypoints only when they are "visible"
   *
   * @see https://github.com/tensorflow/tfjs-models/tree/master/posenet
   *
   * @param {ARR} keypoints The list of all keypoints
   * @param {NUM} minConfidence The minimum keypoint score needed to track
   * @param {OBJ} context The canvas context to draw into
   */
  drawKeypoints(keypoints, minConfidence, context) {
    keypoints.forEach(({ position, score }) => {
      if (score > minConfidence) {
        context.beginPath();
        context.arc(position.x, position.y, 15, 0, 2 * Math.PI);
        context.fillStyle = "#00ff00";
        context.fill();
      }
    });
  }

  /**
   * Draw each tracked skeleton
   * @see https://github.com/tensorflow/tfjs-models/tree/master/posenet
   *
   * - Draws all visible segments captured with PoseNet.getAdjacentKeyPoints
   *
   * @param {ARR} adjacentPoints The list of all keypoints and their relationships
   * @param {OBJ} context The canvas context to draw into
   */
  drawSkeleton(adjacentPoints, context) {
    adjacentPoints.forEach(keypoints => {
      this.drawSegment(
        this.toTuple(keypoints[0].position),
        this.toTuple(keypoints[1].position),
        context
      );
    });
  }

  /**
   * Converts a position to a tuple
   * - Essentially converts an {x, y} object into a [y, x] array
   *
   * @param {OBJ} position {x, y}
   */
  toTuple({ x, y }) {
    return [y, x];
  }

  /**
   * Draws the skeleton segment
   * - A segment is a straight line between two tuples
   *
   * @param {OBJ} fromTuple [ay, ax] The starting point
   * @param {OBJ} toTuple [by, bx] The ending point
   * @param {HEX} color The color to draw in
   * @param {OBJ} context The canvas context to draw in
   */
  drawSegment([ay, ax], [by, bx], context) {
    const scale = 1;

    context.beginPath();
    context.moveTo(ax * scale, ay * scale);
    context.lineTo(bx * scale, by * scale);
    context.lineWidth = 10;
    context.strokeStyle = "#ff00ff";
    context.stroke();
  }
}

require("./methods/calculations")(PosenetPointer);
module.exports = PosenetPointer;
