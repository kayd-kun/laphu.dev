// <!-- Google tag (gtag.js) -->
// Dynamically load the Google Tag Manager script
function loadGTagScript() {
  const script = document.createElement("script");
  script.async = true;
  script.src = "https://www.googletagmanager.com/gtag/js?id=G-0N7L5444P6";
  document.head.appendChild(script);

  // Initialize the dataLayer
  window.dataLayer = window.dataLayer || [];
  function gtag() {
    dataLayer.push(arguments);
  }
  gtag("js", new Date());

  // Configure Google Tag Manager
  gtag("config", "G-0N7L5444P6");
}

// Call the function to load the script
loadGTagScript();
