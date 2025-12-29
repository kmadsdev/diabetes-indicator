document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("surveyForm");
    const bmiType = document.getElementById("bmiType");
    const weightInput = document.getElementById("weightInput");
    const heightInput = document.getElementById("heightInput");
    const bmiHidden = document.getElementById("BMI");
    const bmiResult = document.getElementById("bmiResult");
    const resultDiv = document.getElementById("result");


    // Handle button groups for options (e.g., Yes/No)
    document.querySelectorAll(".buttons").forEach(group => {
        const buttons = group.querySelectorAll("button");
        const hiddenInput = group.nextElementSibling;

        buttons.forEach(btn => {
            btn.addEventListener("click", () => {
                buttons.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                hiddenInput.value = btn.dataset.value;
            });
        });
    });


    // Update placeholders based on metric type
    function updatePlaceholders() {
        const type = bmiType.value;
        if (type === "metric") {
            weightInput.placeholder = "kg";
            heightInput.placeholder = "cm";
        } else {
            weightInput.placeholder = "lbs";
            heightInput.placeholder = "ft";
        }
    }

    // BMI calculation
    function calculateBMI() {
        const weight = parseFloat(weightInput.value);
        const height = parseFloat(heightInput.value);
        const type = bmiType.value;

        if (!weight || !height) {
            bmiResult.textContent = "";
            bmiHidden.value = "";
            return;
        }

        let bmi = type === "metric"
            ? weight / Math.pow(height / 100, 2)
            : (weight / Math.pow(height, 2)) * 703;

        const category =
            bmi < 18.5 ? "Underweight" :
            bmi < 25 ?   "Normal Weight" :
            bmi < 30 ?   "Overweight" : 
            bmi < 35 ?   "Obesity Class I" : 
            bmi < 40 ?   "Obesity Class II" : 
            "Severe Obesity";

        bmiHidden.value = bmi.toFixed(1);
        bmiResult.textContent = `Your BMI is: ${bmi.toFixed(1)} — Category: ${category}`;
    }

    [weightInput, heightInput].forEach(el =>
        el.addEventListener("input", calculateBMI)
    );

    bmiType.addEventListener("change", () => {
        updatePlaceholders();
        calculateBMI();
    });

    // Initialize placeholders
    updatePlaceholders();

    // Form submission
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formData = new FormData(form);

        // Convert all inputs to numeric values
        const inputs = Array.from(formData.values())
            .map(v => v.includes('.') ? parseFloat(v) : parseInt(v))
            .join(",");

        // Show loading state
        resultDiv.className = "result loading";
        resultDiv.innerHTML = '<span class="loading-dots">Analyzing your health data</span>';

        try {
            const response = await fetch(`https://api.kmads.dev/diabetes-indicator/predict?inputs=${inputs}`);
            if (!response.ok) throw new Error("API connection failed");

            const data = await response.json();

            // Remove loading state
            resultDiv.className = "result";
            resultDiv.style.color = data.prediction === 1 ? "#ffff00" : "#00ff00";
            resultDiv.textContent = data.prediction === 1
                ? `⚠️ You may have diabetes (${data.accuracy}% confidence). You might wanna see a doctor.`
                : `✅ You likely don't have diabetes (${data.accuracy}% confidence).`;

        } catch (err) {
            resultDiv.className = "result";
            resultDiv.style.color = "#ff0000";
            resultDiv.textContent = "Error connecting to API.";
            console.error(err);
        }
    });
});
