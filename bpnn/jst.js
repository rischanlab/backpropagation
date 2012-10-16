var _ = require("underscore"),
    lookup = require("./lookup");

var NeuralNetwork = function(options) {
  options = options || {};
  this.learningRate = options.learningRate || 0.3;
  this.momentum = options.momentum || 0.1;
  this.hiddenSizes = options.hiddenLayers;
}

NeuralNetwork.prototype = {
  initialize: function(sizes) {
    this.sizes = sizes;
    this.outputLayer = this.sizes.length - 1;

    this.bias = []; 
    this.bobot = [];
    this.output = [];

 
    this.delta = [];
    this.ganti = []; 
    this.errors = [];

    for (var layer = 0; layer <= this.outputLayer; layer++) {
      var size = this.sizes[layer];
      this.delta[layer] = setnol(size);
      this.errors[layer] = setnol(size);
      this.output[layer] = setnol(size);

      if (layer > 0) {
        this.bias[layer] = setrandom(size);
        this.bobot[layer] = new Array(size);
        this.ganti[layer] = new Array(size);

        for (var node = 0; node < size; node++) {
          var prevSize = this.sizes[layer - 1];
          this.bobot[layer][node] = setrandom(prevSize);
          this.ganti[layer][node] = setnol(prevSize);
        }
      }
    }
  },

  run: function(input) { //untuk testing
    if (this.inputLookup) {
      input = lookup.toArray(this.inputLookup, input);
    }

    var output = this.runInput(input);

    if (this.outputLookup) {
      output = lookup.toHash(this.outputLookup, output);
    }
    return output;
  },

  
  runInput: function(input) {
    this.output[0] = input;  // set output state of input layer

    for (var layer = 1; layer <= this.outputLayer; layer++) {
      for (var node = 0; node < this.sizes[layer]; node++) {
        var bobot = this.bobot[layer][node];
     //menghitung input di hiden layer dan output layer
        var sum = this.bias[layer][node];
        for (var k = 0; k < bobot.length; k++) {
          sum += bobot[k] * input[k];
        }
        this.output[layer][node] = 1 / (1 + Math.exp(-sum));
      }
      var output = input = this.output[layer];
    }
    return output;
  },

  train: function(data, options) {
    data = this.formatData(data);

    options = options || {};
    var iterasi = options.iterasi || 20000;
    var errorThresh = options.errorThresh || 0.005;
    var log = options.log || false;
    var logPeriod = options.logPeriod || 10;
    var callback = options.callback;
    var callbackPeriod = options.callbackPeriod || 10;

    var inputSize = data[0].input.length;
    var outputSize = data[0].output.length;

    var hiddenSizes = this.hiddenSizes;
    if (!hiddenSizes) {
      hiddenSizes = [Math.max(3, Math.floor(inputSize / 2))]; //floor untuk membulatkan bilangan ke integer terbawah
    }
    var sizes = _([inputSize, hiddenSizes, outputSize]).flatten();
    this.initialize(sizes);

    var error = 1;
    for (var i = 0; i < iterasi && error > errorThresh; i++) {
      var sum = 0;
      for (var j = 0; j < data.length; j++) {
        var err = this.trainPattern(data[j].input, data[j].output);
        sum += err;
      }
      error = sum / data.length;

      if (log && (i % logPeriod == 0)) {
        console.log("iterasi:", i, " error training:", error);
      }
      if (callback && (i % callbackPeriod == 0)) {
        callback({ error: error, iterasi: i });
      }
    }

    return {
      error: error,
      iterasi: i
    };
  },

  trainPattern : function(input, target) {
    // feed forwardnya yaitu dari input ke hiden layer kemudian layer output
    this.runInput(input);

    // back propagation .. kembali ke belakang,dan juga update bobot
    this.hitungDelta(target);
    this.updateBobot();

    var error = mse(this.errors[this.outputLayer]); 
    return error; //mengembalikan nilai error yang bisa di tampilkan
  },

  hitungDelta: function(target) {
    for (var layer = this.outputLayer; layer >= 0; layer--) {
      for (var node = 0; node < this.sizes[layer]; node++) {
        var output = this.output[layer][node];

        var error = 0;
        if (layer == this.outputLayer) {
          error = target[node] - output;
        }
        else {
          var delta = this.delta[layer + 1];
          for (var k = 0; k < delta.length; k++) {
            error += delta[k] * this.bobot[layer + 1][k][node];
          }
        }
        this.errors[layer][node] = error;
        this.delta[layer][node] = error * output * (1 - output);
      }
    }
  },

  updateBobot: function() {
    for (var layer = 1; layer <= this.outputLayer; layer++) {
      var incoming = this.output[layer - 1];

      for (var node = 0; node < this.sizes[layer]; node++) {
        var delta = this.delta[layer][node];

        for (var k = 0; k < incoming.length; k++) {
          var change = this.ganti[layer][node][k];

          change = (this.learningRate * delta * incoming[k])
                   + (this.momentum * change);

          this.ganti[layer][node][k] = change;
          this.bobot[layer][node][k] += change;
        }
        this.bias[layer][node] += this.learningRate * delta;
      }
    }
  },

  formatData: function(data) {
    // turn sparse hash input into arrays with 0s as filler
    if (!_(data[0].input).isArray()) {
      if (!this.inputLookup) {
        this.inputLookup = lookup.buildLookup(_(data).pluck("input"));
      }
      data = data.map(function(datum) {
        var array = lookup.toArray(this.inputLookup, datum.input)
        return _(_(datum).clone()).extend({ input: array });
      }, this);
    }

    if (!_(data[0].output).isArray()) {
      if (!this.outputLookup) {
        this.outputLookup = lookup.buildLookup(_(data).pluck("output"));
      }
      data = data.map(function(datum) {
        var array = lookup.toArray(this.outputLookup, datum.output);
        return _(_(datum).clone()).extend({ output: array });
      }, this);
    }
    return data;
  }
}


//fungsi untuk random bilangan 
function randomWeight() {
  return Math.random() * 0.4 - 0.2;
}

//fungsi untuk set bilangan jadi 0 bila kurang dari size
function setnol(size) {
  var array = new Array(size);
  for (var i = 0; i < size; i++) {
    array[i] = 0;
  }
  return array;
}

//funsi untuk set size jadi random weight
function setrandom(size) {
  var array = new Array(size);
  for (var i = 0; i < size; i++) {
    array[i] = randomWeight();
  }
  return array;
}

//Jika MSE sekarang lebih kecil dari MSE sebelumnya, maka laju pembelajaran dinaikkan. 
function mse(errors) {
  // mean squared error
  var sum = 0;
  for (var i = 0; i < errors.length; i++) {
    sum += Math.pow(errors[i], 2);
  }
  return sum / errors.length;
}

exports.NeuralNetwork = NeuralNetwork;
