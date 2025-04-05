document.addEventListener('DOMContentLoaded', function() {
    const headings = document.querySelectorAll('.post h2');
    
    headings.forEach((heading, index) => {
        const card = document.createElement('div');
        card.className = 'h2-card';

        const headingClone = heading.cloneNode(true);
        card.appendChild(headingClone);

        const content = document.createElement('div');
        content.className = 'h2-card-body';

        let sibling = heading.nextElementSibling;
        const siblingElements = [];
    
        while (sibling && sibling.tagName !== 'H2') {
        siblingElements.push(sibling);
        sibling = sibling.nextElementSibling;
        }
    
        siblingElements.forEach(element => {
        content.appendChild(element.cloneNode(true));
        });
    
        card.appendChild(content);
    
        heading.parentNode.replaceChild(card, heading);
        
        siblingElements.forEach(element => {
        if (element.parentNode) {
            element.parentNode.removeChild(element);
        }
        });
    });
});