(function () {
  "use strict";

  var selected = { a: null, b: null };
  var resetSearch = {};

  var predictBtn = document.getElementById("predict-btn");
  var resetBtn = document.getElementById("reset-btn");
  var errorEl = document.getElementById("error");
  var resultsEl = document.getElementById("results");

  function debounce(fn, ms) {
    var t;
    return function () {
      var args = arguments;
      clearTimeout(t);
      t = setTimeout(function () { fn.apply(null, args); }, ms);
    };
  }

  function prefersReducedMotion() {
    return window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function setupSearch(corner) {
    var input = document.getElementById("fighter-" + corner);
    var list = document.getElementById("suggestions-" + corner);
    var picked = document.getElementById("picked-" + corner);

    var fighters = [];
    var activeIndex = -1;

    function optionId(i) {
      return "suggestion-" + corner + "-" + i;
    }

    function closeList() {
      list.hidden = true;
      input.setAttribute("aria-expanded", "false");
      input.removeAttribute("aria-activedescendant");
      activeIndex = -1;
    }

    function setActive(i) {
      var items = list.querySelectorAll("li");
      items.forEach(function (li, idx) {
        li.setAttribute("aria-selected", idx === i ? "true" : "false");
      });
      activeIndex = i;
      if (i >= 0 && items[i]) {
        input.setAttribute("aria-activedescendant", optionId(i));
        items[i].scrollIntoView({ block: "nearest" });
      } else {
        input.removeAttribute("aria-activedescendant");
      }
    }

    function renderList() {
      list.innerHTML = "";
      fighters.forEach(function (f, i) {
        var li = document.createElement("li");
        li.id = optionId(i);
        li.setAttribute("role", "option");
        li.setAttribute("aria-selected", "false");
        var name = document.createElement("strong");
        name.textContent = f.name;
        var rec = document.createElement("span");
        rec.textContent = "UFC " + f.record;
        li.appendChild(name);
        li.appendChild(rec);
        li.addEventListener("mousedown", function (e) {
          e.preventDefault();
          choose(f);
        });
        list.appendChild(li);
      });
      list.hidden = fighters.length === 0;
      input.setAttribute("aria-expanded", fighters.length > 0 ? "true" : "false");
      activeIndex = -1;
    }

    var search = debounce(function (q) {
      if (q.length < 2) { fighters = []; closeList(); return; }
      fetch("/api/fighters?q=" + encodeURIComponent(q))
        .then(function (r) { return r.json(); })
        .then(function (results) {
          fighters = results;
          renderList();
        })
        .catch(function () { fighters = []; closeList(); });
    }, 200);

    function choose(f) {
      selected[corner] = f.name;
      input.value = f.name;
      picked.textContent = "UFC record: " + f.record + " · " + f.fights + " fights";
      fighters = [];
      closeList();
      updateButton();
    }

    input.addEventListener("input", function () {
      selected[corner] = null;
      picked.textContent = "";
      updateButton();
      search(input.value.trim());
    });

    input.addEventListener("keydown", function (e) {
      if (list.hidden || fighters.length === 0) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActive((activeIndex + 1) % fighters.length);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActive((activeIndex - 1 + fighters.length) % fighters.length);
      } else if (e.key === "Enter") {
        if (activeIndex >= 0) {
          e.preventDefault();
          choose(fighters[activeIndex]);
        }
      } else if (e.key === "Escape") {
        fighters = [];
        closeList();
      }
    });

    input.addEventListener("blur", function () {
      setTimeout(function () { fighters = []; closeList(); }, 150);
    });

    resetSearch[corner] = function () {
      fighters = [];
      closeList();
    };
  }

  function updateButton() {
    predictBtn.disabled = !(selected.a && selected.b && selected.a !== selected.b);
  }

  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = false;
  }

  function renderResults(data) {
    var a = data.fighter_a, b = data.fighter_b;
    var pa = Math.round(a.prob * 100), pb = 100 - pa;
    var favorite = a.prob >= b.prob ? a : b;

    document.getElementById("verdict").textContent =
      favorite.name + " is favored to win";
    var fillA = document.getElementById("prob-fill-a");
    fillA.style.width = pa + "%";
    fillA.style.transform = "scaleX(0)";
    requestAnimationFrame(function () {
      fillA.style.transform = "scaleX(1)";
    });
    document.getElementById("prob-label-a").textContent = a.name + " — " + pa + "%";
    document.getElementById("prob-label-b").textContent = pb + "% — " + b.name;
    document.getElementById("col-a").textContent = a.name;
    document.getElementById("col-b").textContent = b.name;

    document.getElementById("card-a").classList.toggle("winner", favorite === a);
    document.getElementById("card-b").classList.toggle("winner", favorite === b);

    var tbody = document.querySelector("#compare-table tbody");
    tbody.innerHTML = "";
    data.comparison.forEach(function (row) {
      var tr = document.createElement("tr");
      var tdA = document.createElement("td");
      tdA.textContent = row.a;
      if (row.favors === "a") tdA.className = "favors";
      else if (row.favors === "b") tdA.className = "trails";
      var tdStat = document.createElement("td");
      tdStat.textContent = row.stat;
      tdStat.className = "stat-name";
      var tdB = document.createElement("td");
      tdB.textContent = row.b;
      if (row.favors === "b") tdB.className = "favors";
      else if (row.favors === "a") tdB.className = "trails";
      tr.appendChild(tdA);
      tr.appendChild(tdStat);
      tr.appendChild(tdB);
      tbody.appendChild(tr);
    });

    resultsEl.hidden = false;
    resultsEl.scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "nearest"
    });
  }

  predictBtn.addEventListener("click", function () {
    errorEl.hidden = true;
    resultsEl.hidden = true;
    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting…";

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fighter_a: selected.a, fighter_b: selected.b })
    })
      .then(function (r) {
        return r.json().then(function (data) {
          if (!r.ok) throw new Error(data.error || "Prediction failed.");
          return data;
        });
      })
      .then(renderResults)
      .catch(function (err) { showError(err.message); })
      .finally(function () {
        predictBtn.textContent = "Predict fight";
        updateButton();
      });
  });

  resetBtn.addEventListener("click", function () {
    ["a", "b"].forEach(function (corner) {
      selected[corner] = null;
      document.getElementById("fighter-" + corner).value = "";
      document.getElementById("picked-" + corner).textContent = "";
      document.getElementById("card-" + corner).classList.remove("winner");
      resetSearch[corner]();
    });
    errorEl.hidden = true;
    resultsEl.hidden = true;
    updateButton();
    document.getElementById("fighter-a").focus();
  });

  setupSearch("a");
  setupSearch("b");
})();
