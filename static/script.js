// script.js

document.addEventListener('DOMContentLoaded', () => {
    const htmlElement = document.documentElement;
    const themeToggle = document.getElementById('theme-toggle');
    const themeIconLight = document.getElementById('theme-icon-light');
    const themeIconDark = document.getElementById('theme-icon-dark');
    const uploadSection = document.getElementById('upload-section');
    const questionSection = document.getElementById('question-section');
    const uploadLoadingSpinner = document.getElementById('upload-loading-spinner');
    const questionLoadingSpinner = document.getElementById('question-loading-spinner');
    const uploadStatus = document.getElementById('upload-status');
    const answerDiv = document.getElementById('answer');

    // --- Theme Toggle Logic ---

    // Function to update icon visibility
    const updateThemeIcons = () => {
        if (htmlElement.classList.contains('dark')) {
            themeIconLight.classList.add('hidden');
            themeIconDark.classList.remove('hidden');
        } else {
            themeIconLight.classList.remove('hidden');
            themeIconDark.classList.add('hidden');
        }
    };

    // Apply saved theme or default to system preference
    const currentTheme = localStorage.getItem('theme');
    if (currentTheme) {
        htmlElement.classList.remove('light', 'dark');
        htmlElement.classList.add(currentTheme);
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        htmlElement.classList.add('dark');
        localStorage.setItem('theme', 'dark');
    } else {
        htmlElement.classList.add('light');
        localStorage.setItem('theme', 'light');
    }
    updateThemeIcons(); // Set initial icon state

    themeToggle.addEventListener('click', () => {
        if (htmlElement.classList.contains('light')) {
            htmlElement.classList.remove('light');
            htmlElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        } else {
            htmlElement.classList.remove('dark');
            htmlElement.classList.add('light');
            localStorage.setItem('theme', 'light');
        }
        updateThemeIcons(); // Update icons on toggle
    });

    // --- HTMX Event Listeners ---

    // Show question section after successful upload
    document.body.addEventListener('htmx:afterSwap', function(event) {
        if (event.detail.target.id === 'upload-status' && event.detail.xhr.status === 200) {
            questionSection.classList.remove('hidden');
            // Clear previous upload status messages and show success
            uploadStatus.innerHTML = '<p class="text-green-600 dark:text-green-400 font-semibold mt-2">PDFs uploaded successfully! You can now ask questions.</p>';
        }
    });

    // Handle loading indicators before a request is sent
    document.body.addEventListener('htmx:beforeRequest', function(event) {
        // HTMX's hx-indicator handles showing the spinner, but we can clear old content here.
        if (event.detail.path === '/upload_pdfs/') {
            uploadStatus.innerHTML = ''; // Clear previous status
        } else if (event.detail.path === '/ask_question/') {
            answerDiv.innerHTML = '<p class="text-gray-500 dark:text-gray-400 animate-pulse">Thinking...</p>'; // Show thinking message
        }
    });

    // Handle responses after a request has completed
    document.body.addEventListener('htmx:afterRequest', function(event) {
        // HTMX's hx-indicator handles hiding the spinner automatically.
        if (event.detail.path === '/upload_pdfs/') {
            if (event.detail.failed) { // More robust check for failure
                uploadStatus.innerHTML = `<p class="error">Error uploading files: ${event.detail.xhr.responseText || 'Please try again.'}</p>`;
            }
        } else if (event.detail.path === '/ask_question/') {
            if (event.detail.failed) {
                answerDiv.innerHTML = `<p class="error">Error getting answer: ${event.detail.xhr.responseText || 'Please try again.'}</p>`;
            } else if (event.detail.xhr.status === 200 && event.detail.target.id === 'answer' && !event.detail.target.innerHTML.trim()) {
                // If no content is returned for the answer, display a default message
                answerDiv.innerHTML = '<p class="text-gray-500 dark:text-gray-400">No specific answer found. Please try rephrasing your question.</p>';
            }
        }
    });
});