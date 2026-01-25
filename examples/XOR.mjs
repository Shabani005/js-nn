import { Layer, NeuralNet } from "../nn.mjs";

const net = new NeuralNet([
  new Layer(2, 2),
  new Layer(2, 1)
]);

net.fit(
  [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ],
  [0, 1, 1, 0]
);

console.log("XOR {0, 0} = ", net.predict([0, 0]));
console.log("XOR {0, 1} = ", net.predict([0, 1]));
console.log("XOR {1, 0} = ", net.predict([1, 0]));
console.log("XOR {1, 1} = ",net.predict([1, 1]));

net.writeJSON("xor.json");
