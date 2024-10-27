document.querySelector("button[type='submit']").addEventListener("mouseover", function() {
    document.body.classList.add("darken");
});

document.querySelector("button[type='submit']").addEventListener("mouseout", function() {
    document.body.classList.remove("darken");
});

document.addEventListener("DOMContentLoaded", function() {
    var links = document.querySelectorAll(".load-link");
    var loading = document.getElementById("loading");

    links.forEach(function(link) {
        link.addEventListener("click", function(event) {
            loading.style.display = "block";
            setTimeout(function() {
                window.location.action = link.action;
            }, 1000);
        });
    });
});

document.addEventListener("DOMContentLoaded", function() {
    document.querySelector("form").addEventListener("submit", function(event) {
        event.preventDefault();
        alert("Python Application was opened....Please Check the Taskbar.");
        setTimeout(function() {
            if (document.querySelector(".alert")) {
                document.querySelector(".alert").remove();
            }
        }, 2000);
        this.submit();
    });
});

