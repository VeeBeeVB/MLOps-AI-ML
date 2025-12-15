document.getElementById('predictionForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const features = [];
    for (let i = 0; i < 10; i++) {
        const value = parseFloat(document.getElementById(`feature${i}`).value);
        if (isNaN(value)) {
            showResult('Please enter valid numbers for all features.', 'error');
            return;
        }
        features.push(value);
    }

    const payload = {
        features: [features]
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        // Assuming the response has predictions array
        const prediction = data.predictions ? data.predictions[0] : data;
        showResult(`Predicted diabetes progression: ${prediction}`, 'success');
    } catch (error) {
        showResult(`Error: ${error.message}`, 'error');
    }
});

function showResult(message, type) {
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = message;
    resultDiv.className = `result ${type}`;
}
