import fs from "fs";
import mnist from "mnist";
import { NeuralNet } from "../nn.mjs";

// Load saved network JSON
const raw = fs.readFileSync("MNIST.json", "utf8");
const modelData = JSON.parse(raw);

// Restore network
const net = new NeuralNet([]).fromJSON(modelData);

// Load MNIST test set
const { test } = mnist.set(0, 200);

let correct = 0;

for (const sample of test) {
  const prediction = net.predict(sample.input);
  const predictedLabel = prediction.indexOf(
    Math.max(...prediction)
  );
  const actualLabel = sample.output.indexOf(1);

  if (predictedLabel === actualLabel) {
    correct += 1;
  }
}

const accuracy = (correct / test.length) * 100;

console.log(
  `Loaded model test accuracy: ${accuracy.toFixed(2)}%`
);
