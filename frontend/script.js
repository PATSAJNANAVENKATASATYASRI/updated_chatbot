document.addEventListener('DOMContentLoaded', () => {
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        messageDiv.innerHTML = text; // Changed from textContent to innerHTML
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function getRoute(origin, destination) {
        try {
            const response = await fetch("http://localhost:8000/route", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ origin, destination })
            });

            const data = await response.json();

            const link = document.createElement("a");
            link.href = data.map_url;
            link.target = "_blank";
            link.textContent = "👉 Click here to view the route on Google Maps";

            const messageDiv = document.createElement("div");
            messageDiv.classList.add("message", "ai-message");
            messageDiv.appendChild(link);

            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;

        } catch (error) {
            addMessage("Could not fetch route. Try again.", "ai");
        }
    }

    async function sendMessage() {
        const query = userInput.value.trim();
        if (query === '') return;

        addMessage(query, 'user');
        userInput.value = '';

        addMessage('Assistant is typing...', 'ai');

        try {
            const routePattern = /from (.+) to (.+)/i;
            const match = query.match(routePattern);

            if (match) {
                chatMessages.removeChild(chatMessages.lastChild);
                const origin = match[1].trim();
                const destination = match[2].trim();
                addMessage(`Finding best route from ${origin} to ${destination}...`, "ai");
                await getRoute(origin, destination);
                return;
            }

            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            chatMessages.removeChild(chatMessages.lastChild);
            addMessage(data.response, 'ai');

        } catch (error) {
            console.error('Error sending message:', error);
            chatMessages.removeChild(chatMessages.lastChild);
            addMessage('Oops! Something went wrong. Please try again.', 'ai');
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });
     addMessage('Hello! How can I assist you with information about Sri Vasavi Engineering College?', 'ai');
});
