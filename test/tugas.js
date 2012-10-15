
ch_ann = require("../bpnn/ch_ann.js");
var net = new ch_ann.NeuralNetwork({
   hiddenLayers: [3], // set hidden layer dan learning rate
   learningRate: 0.6
});



net.train([{input: [0,1,0,1,1,1,0,1,0], output: { tambah: 1 }}, 
           {input: [0,0,0,1,1,1,0,0,0], output: { kurang: 1 }},
           {input: [1,0,1,0,1,0,1,0,1], output: { kali: 1 }}], {
  errorThresh: 0.004,  // error treshold
  iterasi: 20000,   // maximum iterasi untuk training
  log: true,           // 
  logPeriod: 1        // tampil per iterasi ? 
});

var output = net.run([1,0,1,0,1,0,1,0,1]);  // kali  0.9371652425075219

console.log(output);