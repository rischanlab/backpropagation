
ch_ann = require("../bpnn/ch_ann.js");
var net = new ch_ann.NeuralNetwork({
   hiddenLayers: [3],
   learningRate: 0.6
});

net.train([{input: [0, 0], output: [0]},
           {input: [0, 1], output: [1]},
           {input: [1, 0], output: [1]},
           {input: [1, 1], output: [0]}], {
  errorThresh: 0.004,  // error threshold to reach
  iterasi: 20000,   // maximum training iterations
  log: true,           // console.log() progress periodically
  logPeriod: 1        // number of iterations between logging
});

var output = net.run([1, 0]);  // [0.987]

console.log(output);