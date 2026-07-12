(function () {
  "use strict";

  var SCRIPT_NAME = "version-switcher.js";

  function currentScriptUrl() {
    if (document.currentScript && document.currentScript.src) {
      return new URL(document.currentScript.src, window.location.href);
    }

    var scripts = document.getElementsByTagName("script");
    for (var i = scripts.length - 1; i >= 0; i -= 1) {
      if (scripts[i].src && scripts[i].src.indexOf(SCRIPT_NAME) !== -1) {
        return new URL(scripts[i].src, window.location.href);
      }
    }

    return null;
  }

  function siteBaseUrl() {
    var scriptUrl = currentScriptUrl();
    if (!scriptUrl) {
      return new URL("./", window.location.href);
    }

    return new URL("../", scriptUrl);
  }

  function currentVersion(baseUrl, versions) {
    var basePath = baseUrl.pathname;
    var currentPath = window.location.pathname;

    if (currentPath.indexOf(basePath) !== 0) {
      return "";
    }

    var relativePath = currentPath.slice(basePath.length);
    var candidate = relativePath.split("/")[0];
    if (versions.indexOf(candidate) !== -1) {
      return candidate;
    }

    return "";
  }

  function relativePagePath(baseUrl, version) {
    var basePath = baseUrl.pathname;
    var currentPath = window.location.pathname;

    if (!version || currentPath.indexOf(basePath) !== 0) {
      return "";
    }

    var relativePath = currentPath.slice(basePath.length);
    var prefix = version + "/";
    if (relativePath.indexOf(prefix) !== 0) {
      return "";
    }

    return relativePath.slice(prefix.length);
  }

  function versionUrl(baseUrl, version, pagePath) {
    if (version === "stable") {
      return new URL("stable/" + window.location.search + window.location.hash, baseUrl).href;
    }

    return new URL(version + "/" + pagePath + window.location.search + window.location.hash, baseUrl).href;
  }

  function createItem(baseUrl, version, activeVersion, pagePath, stableVersion) {
    var item = document.createElement("a");
    item.className = "he-doc-version-switcher__item";
    item.href = versionUrl(baseUrl, version, pagePath);
    item.textContent = version;

    if (version === activeVersion) {
      item.setAttribute("aria-current", "page");
    }

    if (version === stableVersion) {
      var badge = document.createElement("span");
      badge.className = "he-doc-version-switcher__badge";
      badge.textContent = "stable";
      item.appendChild(badge);
    }

    return item;
  }

  function render(baseUrl, metadata) {
    var versions = Array.isArray(metadata.versions) ? metadata.versions : [];
    if (versions.length === 0 || document.querySelector(".he-doc-version-switcher")) {
      return;
    }

    var activeVersion = currentVersion(baseUrl, versions);
    var pagePath = relativePagePath(baseUrl, activeVersion);
    var container = document.createElement("nav");
    var toggle = document.createElement("button");
    var label = document.createElement("span");
    var current = document.createElement("span");
    var chevron = document.createElement("span");
    var panel = document.createElement("div");

    container.className = "he-doc-version-switcher";
    container.setAttribute("aria-label", "Documentation versions");

    toggle.className = "he-doc-version-switcher__toggle";
    toggle.type = "button";
    toggle.setAttribute("aria-haspopup", "true");
    toggle.setAttribute("aria-expanded", "false");

    label.className = "he-doc-version-switcher__label";
    label.textContent = "Version";

    current.className = "he-doc-version-switcher__current";
    current.textContent = activeVersion || metadata.stable || versions[0];

    chevron.className = "he-doc-version-switcher__chevron";
    chevron.setAttribute("aria-hidden", "true");
    chevron.textContent = "^";

    panel.className = "he-doc-version-switcher__panel";

    versions.forEach(function (version) {
      panel.appendChild(createItem(baseUrl, version, activeVersion, pagePath, metadata.stable));
    });

    toggle.appendChild(label);
    toggle.appendChild(current);
    toggle.appendChild(chevron);
    container.appendChild(toggle);
    container.appendChild(panel);
    document.body.appendChild(container);

    toggle.addEventListener("click", function () {
      var isOpen = container.getAttribute("data-open") === "true";
      container.setAttribute("data-open", isOpen ? "false" : "true");
      toggle.setAttribute("aria-expanded", isOpen ? "false" : "true");
    });

    document.addEventListener("click", function (event) {
      if (!container.contains(event.target)) {
        container.setAttribute("data-open", "false");
        toggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  function init() {
    var baseUrl = siteBaseUrl();
    fetch(new URL("versions.json", baseUrl).href, { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("Unable to load documentation versions.");
        }
        return response.json();
      })
      .then(function (metadata) {
        render(baseUrl, metadata);
      })
      .catch(function () {
        return undefined;
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
