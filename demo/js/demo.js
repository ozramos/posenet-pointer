/**
 * Sets up the demo projects
 */
const $start = document.querySelector("#start-posenet");
const $selector = document.querySelector("#demo-selector");
window.$pointer = document.querySelector(".posenet-pointer");

const demo = {
  "simple-pointer": require("./simple-pointer.js").default
};

$start.addEventListener("click", () => {
  demo[$selector.value].start();
});
