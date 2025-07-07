// ui.js
function updateProbO(po) {
    const probO = document.getElementById('prob-o');
    const probOVal = document.getElementById('prob-o-val');
    probO.style.width = `${po}%`;
    probOVal.textContent = `${po}%`;
}
function render(b, t, over, msg, px=50, po=50) {
    board = b;
    turn = t;
    gameOver = over;
    status.textContent = msg;
    boardDiv.innerHTML = '';
    updateProbO(po);
    for (let i = 0; i < 9; i++) {
        let cell = document.createElement('div');
        cell.className = 'cell';
        if (board[i]) cell.classList.add('filled');
        cell.textContent = board[i];
        cell.onclick = () => {
            if (!gameOver && board[i] === "") {
                fetch(`/move?pos=${i}`, {method: 'POST'})
                .then(r => r.json())
                .then(data => render(data.board, data.turn, data.game_over, data.status, data.prob_x, data.prob_o));
            }
        };
        boardDiv.appendChild(cell);
    }
}
function setupUI() {
    window.board = Array(9).fill("");
    window.turn = "X";
    window.gameOver = false;
    window.status = document.getElementById('status');
    window.boardDiv = document.getElementById('board');
    window.resetBtn = document.getElementById('reset');
    resetBtn.onclick = () => {
        fetch('/reset', {method: 'POST'})
        .then(r => r.json())
        .then(data => render(data.board, data.turn, data.game_over, data.status, data.prob_x, data.prob_o));
    };
    fetch('/state').then(r => r.json()).then(data => render(data.board, data.turn, data.game_over, data.status, data.prob_x, data.prob_o));
}
window.onload = setupUI;
