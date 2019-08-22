import * as Posenet from "@tensorflow-models/posenet";
import { merge } from "lodash";

module.exports = PosenetPointer => {
  /**
   * Creates a default (flipped) video and adds it to the DOM:
   * - The video is absolutely positioned within the $wrap
   *
   * @param {HTMLElement} $wrap A container to embed the video into
   *
   * @return {HTMLVideoElement} A hidden video used for inference with PoseNet
   */
  PosenetPointer.prototype.createDefaultVideo = function($wrap) {
    const $video = document.createElement("video");

    $wrap.classList.add("posenet-pointer-debug");
    $video.setAttribute("playsinline", true);
    $video.style.transform = "scale(-1, 1)";
    $video.style.position = "absolute";
    $wrap.appendChild($video);

    return $video;
  };

  /**
   * Creates a default (flipped) canvas and adds it to the DOM
   * - The canvas is added to the $wrap (along with the video) relatively
   *
   * @param {Element} $wrap The wrapping element to inject the canvas into
   *
   * @return {HTMLCanvasElement} A hidden canvas used for debugging with PoseNet
   */
  PosenetPointer.prototype.createDefaultCanvas = function($wrap) {
    const $canvas = document.createElement("canvas");
    $canvas.style.transform = "scale(-1, 1)";
    $canvas.style.position = "relative";
    $canvas.style.top = 0;
    $canvas.style.left = 0;

    $wrap.appendChild($canvas);

    return $canvas;
  };

  /**
   * Helpers for checking if we're on mobile
   * - Checks if we're on mobile
   * - Checks if we're on android
   * - Checks if we're on iOS
   */
  PosenetPointer.prototype.isMobile = function() {
    return this.isAndroid() || this.isIOS();
  };
  PosenetPointer.prototype.isAndroid = function() {
    return /Android/i.test(navigator.userAgent);
  };
  PosenetPointer.prototype.isIOS = function() {
    return /iPhone|iPad|iPod/i.test(navigator.userAgent);
  };

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
  PosenetPointer.prototype.isWebGLSupported = function() {
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
  };

  /**
   * Sets up the webcam and stream:
   * - This creates its own video/canvas elements, allowing you to have
   *    multiple instances going (for example, to use front/back cameras
   *    simultaneously)
   * - Recreates the video feed to reassign srcObject once it's been stopped
   */
  PosenetPointer.prototype.setupFeed = async function() {
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
  };

  /**
   * Init PoseNet
   */
  PosenetPointer.prototype.initPosenet = async function() {
    if (!this.posenet) {
      this.posenet = await Posenet.load({
        architecture: this.options.posenet.architecture,
        multiplier: this.options.posenet.multiplier,
        outputStride: this.options.posenet.outputStride
      });
    }
  };

  /**
   * Sets defaults to the missing constructor options:
   * - Sets defaults
   * - Creates a default debug container
   * - Creates a default video and canvas
   *
   * @param {Object} opts The options passed into the constructor. Pass null to use all defaults
   */
  PosenetPointer.prototype.configure = async function(opts) {
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
  };
};
