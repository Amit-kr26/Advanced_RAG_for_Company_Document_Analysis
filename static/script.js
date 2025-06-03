// script.js

document.addEventListener('DOMContentLoaded', () => {
    const htmlElement = document.documentElement;
    const themeToggle = document.getElementById('theme-toggle');
    const uploadSection = document.getElementById('upload-section');
    const questionSection = document.getElementById('question-section');
    const uploadLoadingSpinner = document.getElementById('upload-loading-spinner');
    const questionLoadingSpinner = document.getElementById('question-loading-spinner');
    const uploadStatus = document.getElementById('upload-status');
    const answerDiv = document.getElementById('answer');

    // --- Theme Toggle Logic ---
    const currentTheme = localStorage.getItem('theme');

    // Apply saved theme or default to system preference
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
        if (event.detail.path === '/upload_pdfs/') {
            uploadLoadingSpinner.classList.remove('hidden');
            uploadStatus.innerHTML = ''; // Clear previous status
        } else if (event.detail.path === '/ask_question/') {
            questionLoadingSpinner.classList.remove('hidden');
            answerDiv.innerHTML = '<p class="text-gray-500 dark:text-gray-400 animate-pulse">Thinking...</p>'; // Show thinking message
        }
    });

    // Handle loading indicators after a request has completed
    document.body.addEventListener('htmx:afterRequest', function(event) {
        if (event.detail.path === '/upload_pdfs/') {
            uploadLoadingSpinner.classList.add('hidden');
            if (event.detail.xhr.status >= 400) {
                uploadStatus.innerHTML = `<p class="error">Error uploading files: ${event.detail.xhr.responseText || 'Please try again.'}</p>`;
            }
        } else if (event.detail.path === '/ask_question/') {
            questionLoadingSpinner.classList.add('hidden');
            if (event.detail.xhr.status >= 400) {
                answerDiv.innerHTML = `<p class="error">Error getting answer: ${event.detail.xhr.responseText || 'Please try again.'}</p>`;
            } else if (event.detail.xhr.status === 200 && event.detail.target.id === 'answer' && !event.detail.target.innerHTML.trim()) {
                // If no content is returned for the answer, display a default message
                answerDiv.innerHTML = '<p class="text-gray-500 dark:text-gray-400">No specific answer found. Please try rephrasing your question.</p>';
            }
        }
    });
});
