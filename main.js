// Main JavaScript for Credit Card Fraud Detection Project

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Form validation for the fraud detection form
    const fraudDetectionForm = document.getElementById('fraudDetectionForm');
    if (fraudDetectionForm) {
        fraudDetectionForm.addEventListener('submit', function(event) {
            // Add loading state
            document.querySelector('.card-body').classList.add('loading');
            
            // Basic form validation
            const inputs = fraudDetectionForm.querySelectorAll('input[required]');
            let isValid = true;
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    isValid = false;
                    input.classList.add('is-invalid');
                } else {
                    input.classList.remove('is-invalid');
                }
            });
            
            if (!isValid) {
                event.preventDefault();
                document.querySelector('.card-body').classList.remove('loading');
                alert('Please fill in all required fields.');
            }
        });
        
        // Clear validation styling on input
        const inputs = fraudDetectionForm.querySelectorAll('input');
        inputs.forEach(input => {
            input.addEventListener('input', function() {
                this.classList.remove('is-invalid');
            });
        });
    }
    
    // Enable tooltips everywhere
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Enable popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Animated counters for metrics on home page
    const animateCounters = () => {
        const counters = document.querySelectorAll('.counter');
        
        counters.forEach(counter => {
            const target = +counter.getAttribute('data-target');
            const duration = 1500; // Animation duration in milliseconds
            const step = target / (duration / 16); // 60fps
            
            let current = 0;
            const updateCounter = () => {
                current += step;
                if (current < target) {
                    counter.textContent = Math.round(current);
                    requestAnimationFrame(updateCounter);
                } else {
                    counter.textContent = target;
                }
            };
            
            updateCounter();
        });
    };
    
    // Run counter animation if elements exist
    if (document.querySelector('.counter')) {
        animateCounters();
    }
    
    // Handle visualization modal zoom
    const plotImages = document.querySelectorAll('.plot-img');
    plotImages.forEach(img => {
        img.addEventListener('click', function() {
            const modalId = this.getAttribute('data-bs-target');
            const modal = document.querySelector(modalId);
            if (modal) {
                const modalImg = modal.querySelector('.modal-img');
                modalImg.src = this.src;
            }
        });
    });
    
    // API demo functionality
    const apiDemoForm = document.getElementById('apiDemoForm');
    if (apiDemoForm) {
        apiDemoForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            // Get JSON input
            const jsonInput = document.getElementById('apiJsonInput').value;
            let jsonData;
            
            try {
                jsonData = JSON.parse(jsonInput);
            } catch (error) {
                alert('Invalid JSON format. Please check your input.');
                return;
            }
            
            // Show loading state
            const resultContainer = document.getElementById('apiResultContainer');
            resultContainer.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-2">Processing request...</p></div>';
            
            // Make API request
            fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: jsonInput,
            })
            .then(response => response.json())
            .then(data => {
                // Format and display the result
                const resultHtml = `
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h5 class="mb-0">API Response</h5>
                        </div>
                        <div class="card-body">
                            <pre class="language-json"><code>${JSON.stringify(data, null, 2)}</code></pre>
                        </div>
                    </div>
                `;
                resultContainer.innerHTML = resultHtml;
                
                // Initialize syntax highlighting if Prism.js is available
                if (typeof Prism !== 'undefined') {
                    Prism.highlightAll();
                }
            })
            .catch(error => {
                resultContainer.innerHTML = `
                    <div class="alert alert-danger">
                        <h5>Error</h5>
                        <p>${error.message}</p>
                    </div>
                `;
            });
        });
    }
});

// Function to generate random transaction data for demo purposes
function generateRandomTransaction() {
    // Generate random values for V1-V28
    const transaction = {
        'Time': Math.floor(Math.random() * 172800), // Random time within 2 days (in seconds)
        'Amount': (Math.random() * 2000).toFixed(2), // Random amount between 0 and 2000
    };
    
    // Generate V1-V28 with values typically between -3 and 3
    for (let i = 1; i <= 28; i++) {
        transaction[`V${i}`] = (Math.random() * 6 - 3).toFixed(6);
    }
    
    return transaction;
}

// Function to fill form with random transaction data
function fillRandomTransaction() {
    const transaction = generateRandomTransaction();
    
    // Fill the form fields
    for (const [key, value] of Object.entries(transaction)) {
        const input = document.getElementById(key);
        if (input) {
            input.value = value;
        }
    }
}

// Function to toggle between form sections in the demo
function toggleFormSection(sectionId) {
    const sections = document.querySelectorAll('.feature-group');
    sections.forEach(section => {
        section.style.display = 'none';
    });
    
    document.getElementById(sectionId).style.display = 'block';
}
