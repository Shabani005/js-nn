class Neuron {
  constructor(input_len, learning_rate=0.1){
    this.learning_rate = learning_rate;
    this.weights = Array(input_len).fill(0);
    this.bias = 0;
  }

  sigmoid(z){
    return 1 / (1 + Math.exp(-z));
  }

  predict(inputs){
    let z = this.bias;

    for (let i =0; i<this.weights.length; i+=1){
      z+= this.weights[i] * inputs[i];
    }
    return this.sigmoid(z);
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
}

const X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]

const y = [0, 0, 0, 1];

const neuron = new Neuron(X[0].length);

neuron.fit(X, y);

console.log(neuron.predict([1, 1]));
console.log(neuron.predict([0, 0]));
