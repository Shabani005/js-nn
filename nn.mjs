import fs from "fs";

export class Neuron {
  constructor(input_len, learning_rate=0.1){
    this.learning_rate = learning_rate;
    this.weights = Array(input_len).fill(0).map(() => Math.random() * 2-1);
    this.bias = 0;
  }

  sigmoid(z){
    return 1 / (1 + Math.exp(-z));
  }
  
  sigmoid_deriv(z){
    return z * (1 - z);
  }

  predict(inputs){
    this.inputs = inputs;
    let z = this.bias;

    for (let i =0; i<this.weights.length; i+=1){
      z+= this.weights[i] * inputs[i];
    }
    this.output = this.sigmoid(z);
    return this.output;
  }

  train(inputs, target){
    const output = this.predict(inputs);

    const error = output-target;

    for (let i=0; i<this.weights.length; i+=1){
      this.weights[i] -= this.learning_rate * error * inputs[i];
    }
    this.bias -= this.learning_rate * error;
  }

  fit(training_data, labels, epochs=1000){
    for (let epoch=0; epoch<epochs; epoch+=1){
      for (let i=0; i<training_data.length; i+=1){
        this.train(training_data[i], labels[i]);
      }
    }
  }

  backward(err){
    const delta = err * this.sigmoid_deriv(this.output);

    for (let i=0; i<this.weights.length; i+=1){
      this.weights[i] -= this.learning_rate * delta * this.inputs[i];
    }
    this.bias -= this.learning_rate * delta;

    return delta;
  }

  toJSON() {
    return {
      weights: this.weights,
      bias: this.bias
    };
  }

  fromJSON(data, learning_rate=0.1){
    const neuron = new Neuron(data.weights.length, learning_rate);
    neuron.weights = data.weights;
    neuron.bias = data.bias;
    return neuron;
  }
}

export class Layer {
  constructor(input_size, neuron_count, learning_rate=0.1){
    this.input_size = input_size;
    this.neurons = Array.from(
      { length: neuron_count },
      () => new Neuron(input_size, learning_rate)
    );
  }

  forward(inputs){
    return this.neurons.map(neuron => neuron.predict(inputs));
  }

  backward(errs){
    // const errs = Array(this.neurons[0].weights.length).fill(0);
    const prev_errs = Array(this.input_size).fill(0);

    for (let i=0; i<this.neurons.length; i+=1){
      const delta = this.neurons[i].backward(errs[i]);

      for (let j=0; j<this.neurons[i].weights.length; j+=1){
        prev_errs[j] += this.neurons[i].weights[j] * delta;
      }
    }
    return prev_errs;
  }

  toJSON() {
    return {
      input_size: this.input_size,
      neurons: this.neurons.map((neuron) => neuron.toJSON())
    };
  }

  fromJSON(data, learning_rate=0.1) {
    const layer = new Layer(
      data.input_size,
      data.neurons.length,
      learning_rate
    );

    layer.neurons = data.neurons.map(neuron => neuron.fromJSON(learning_rate));

    return layer;
  }
}

export class NeuralNet {
  constructor(layers){
    this.layers = layers;
  }

  predict(inputs){
    let output = inputs;

    for (const layer of this.layers){
      output = layer.forward(output);
    }
    return output;
  }

  train(inputs, targets){
    const outputs = this.predict(inputs);

    let errs = outputs.map(
      (output, i) => output - targets[i]
    );

    for (let i=this.layers.length-1; i>=0; i-=1){
      errs = this.layers[i].backward(errs);
    }
  }

  fit(X, y, epochs=10000){
    for (let epoch=0; epoch<epochs; epoch+=1){
      for (let i=0; i<X.length; ++i){
        this.train(X[i], [y[i]]);
      }
    }
  }

  toJSON() {
    return {
      layers: this.layers.map(layer => layer.toJSON())
    };
  }

  fromJSON(data, learning_rate=0.1) {
    const layers = data.layers.map(layer => layer.fromJSON(learning_rate));
    return new NeuralNet(layers);
  };
  
  writeJSON(file_name){
    const nnet_json = JSON.stringify(this.toJSON(), null, 2);
    fs.writeFileSync(file_name, nnet_json, "utf-8");
  }
}

// module.exports { Neuron, Layer, NeuralNet };

