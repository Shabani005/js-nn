import { NeuralNet, Layer } from "../nn.mjs";
import mnist from "mnist";

const { training, test } = mnist.set(800, 200);

const net = new NeuralNet([
  new Layer(784, 64, 0.05),
  new Layer(64, 32, 0.05),
  new Layer(32, 10, 0.05)
]);

console.log("Training...");

for (let epoch = 0; epoch < 20; epoch += 1) {
  let correct = 0;

  for (const sample of training) {
    net.train(sample.input, sample.output);

    const prediction = net.predict(sample.input);
    const predictedLabel = prediction.indexOf(
      Math.max(...prediction)
    );
    const actualLabel = sample.output.indexOf(1);

    if (predictedLabel === actualLabel) {
      correct += 1;
    }
  }

  const accuracy = (correct / training.length) * 100;
  console.log(
    `Epoch ${epoch + 1}: Training accuracy = ${accuracy.toFixed(2)}%`
  );
}

let testCorrect = 0;

for (const sample of test) {
  const prediction = net.predict(sample.input);
  const predictedLabel = prediction.indexOf(
    Math.max(...prediction)
  );
  const actualLabel = sample.output.indexOf(1);

  if (predictedLabel === actualLabel) {
    testCorrect += 1;
  }
}

const testAccuracy = (testCorrect / test.length) * 100;

console.log(
  `Test accuracy: ${testAccuracy.toFixed(2)}%`
);

net.writeJSON("MNIST.json");
