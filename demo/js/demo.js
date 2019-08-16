/**
 * Sets up the demo projects
 */
const demo = {
  noConfig: require("./simple-pointer.js").default
};
const $start = document.querySelector("#start-posenet");
const $selector = document.querySelector("#demo-selector");
window.$pointer = document.querySelector(".posenet-pointer");

$start.addEventListener("click", () => {
  demo[$selector.value].start();
});
