(function () {
  "use strict";

  var selected = { a: null, b: null };

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

  function setupSearch(corner) {
    var input = document.getElementById("fighter-" + corner);
    var list = document.getElementById("suggestions-" + corner);
    var picked = document.getElementById("picked-" + corner);

    var search = debounce(function (q) {
      if (q.length < 2) { list.hidden = true; return; }
      fetch("/api/fighters?q=" + encodeURIComponent(q))
        .then(function (r) { return r.json(); })
        .then(function (fighters) {
          list.innerHTML = "";
          if (!fighters.length) { list.hidden = true; return; }
          fighters.forEach(function (f) {
            var li = document.createElement("li");
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
          list.hidden = false;
        })
        .catch(function () { list.hidden = true; });
    }, 200);

    function choose(f) {
      selected[corner] = f.name;
      input.value = f.name;
      picked.textContent = "UFC record: " + f.record + " · " + f.fights + " fights";
      list.hidden = true;
      updateButton();
    }

    input.addEventListener("input", function () {
      selected[corner] = null;
      picked.textContent = "";
      updateButton();
      search(input.value.trim());
    });
    input.addEventListener("blur", function () {
      setTimeout(function () { list.hidden = true; }, 150);
    });
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
    document.getElementById("prob-fill").style.width = pa + "%";
    document.getElementById("prob-label-a").textContent = a.name + " — " + pa + "%";
    document.getElementById("prob-label-b").textContent = pb + "% — " + b.name;
    document.getElementById("col-a").textContent = a.name;
    document.getElementById("col-b").textContent = b.name;

    var tbody = document.querySelector("#compare-table tbody");
    tbody.innerHTML = "";
    data.comparison.forEach(function (row) {
      var tr = document.createElement("tr");
      var tdA = document.createElement("td");
      tdA.textContent = row.a;
      if (row.favors === "a") tdA.className = "favors";
      var tdStat = document.createElement("td");
      tdStat.textContent = row.stat;
      tdStat.className = "stat-name";
      var tdB = document.createElement("td");
      tdB.textContent = row.b;
      if (row.favors === "b") tdB.className = "favors";
      tr.appendChild(tdA);
      tr.appendChild(tdStat);
      tr.appendChild(tdB);
      tbody.appendChild(tr);
    });

    resultsEl.hidden = false;
    resultsEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
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
    });
    errorEl.hidden = true;
    resultsEl.hidden = true;
    updateButton();
  });

  setupSearch("a");
  setupSearch("b");
})();
