$(document).ready(function () {
    let currentUserId = null;

    // Fetch current user ID on page load
    fetchUserId();

    function fetchUserId() {
        $.ajax({
            url: "/get_current_user",
            type: "GET",
            success: function (response) {
                if (response.user_id) {
                    currentUserId = response.user_id;
                } else {
                    alert("User not logged in!");
                    window.location.href = "/login";
                }
            }
        });
    }

    // Fetch stock price when symbol is entered
    $("#symbol").on("change", function () {
        let symbol = $(this).val().toUpperCase();
        if (symbol) {
            fetchStockPrice(symbol);
        }
    });

    function fetchStockPrice(symbol) {
        $.ajax({
            url: `/get_stock_price?symbol=${symbol}`,
            type: "GET",
            success: function (response) {
                if (response.price) {
                    $("#price").val(response.price.toFixed(2));
                } else {
                    $("#price").val("Invalid Symbol");
                }
            },
            error: function () {
                $("#price").val("Error fetching price");
            }
        });
    }

    // Handle Buy & Sell actions
    $("#buy-btn, #sell-btn").on("click", function (event) {
        event.preventDefault();

        let symbol = $("#symbol").val().toUpperCase();
        let shares = $("#shares").val();
        let price = $("#price").val();
        let transactionType = $(this).attr("id") === "buy-btn" ? "BUY" : "SELL";

        if (!symbol || !shares || !price || isNaN(shares) || shares <= 0 || isNaN(price)) {
            showMessage("error", "Please enter valid stock details.");
            return;
        }

        if (!currentUserId) {
            showMessage("error", "User not authenticated. Please log in.");
            return;
        }

        $.ajax({
            url: transactionType === "BUY" ? "/buy_stock" : "/sell_stock",
            type: "POST",
            data: {
                user_id: currentUserId,
                symbol: symbol,
                shares: shares,
                price: price
            },
            success: function (response) {
                showMessage("success", response.message);
            },
            error: function (xhr) {
                let errorMsg = xhr.responseJSON?.error || "Transaction failed.";
                showMessage("error", errorMsg);
            }
        });
    });

    function showMessage(type, message) {
        if (type === "success") {
            $("#success-message").text(message).fadeIn();
            $("#error-message").hide();
        } else {
            $("#error-message").text(message).fadeIn();
            $("#success-message").hide();
        }

        setTimeout(() => {
            $(".message").fadeOut();
        }, 3000);
    }
});
