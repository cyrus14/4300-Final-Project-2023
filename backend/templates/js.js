function randomNumber(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min
}

const STAR_COUNT = 100
let result = ""
for(let i = 0; i < STAR_COUNT; i++){
    result += `${randomNumber(-50, 50)}vw ${randomNumber(-50, 50)}vh ${randomNumber(0, 3)}px ${randomNumber(0, 3)}px #fff,`
}
console.log(result.substring(0, result.length - 1))

const input = document.querySelector("input[type='range']");
for (const event of ["input", "change"])
	input.addEventListener(event, (e) => update(e.target));

function update(input) {
	for (const data of ["min", "max", "value"])
		if (input[data]) input.style.setProperty(`--${data}`, input[data]);
}

update(input);