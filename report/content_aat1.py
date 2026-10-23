"""AAT 1: Perform basic tensor operations using TensorFlow."""
from docx_helpers import *

F = "report/figures/"


def build(doc, figno):
    title(doc, "Perform Basic Tensor Operations using TensorFlow")

    heading(doc, "1. Introduction")
    para(doc, "This experiment creates tensors of rank 0 to 3 in TensorFlow 2.20 and applies eleven basic operations to "
              "them: addition, subtraction, element-wise multiplication, matrix multiplication, division, transpose, "
              "reshape, square and three reductions. Every result was checked by hand and matches the expected value.")
    para(doc, "A tensor is a multi-dimensional array of numbers and is the basic data structure of every deep learning "
              "framework. An image is a 3D tensor (height, width, channels), a batch of images is a 4D tensor, and the "
              "weights of a layer are a 2D tensor. Training a neural network is, at its core, a long chain of tensor "
              "operations: the forward pass multiplies inputs by weights and adds a bias, and the backward pass "
              "computes gradients of these same operations.")
    para(doc, "TensorFlow stores data as tensors instead of plain Python lists because tensors can run on a GPU, support "
              "automatic differentiation and apply one operation to every element at once (vectorisation). Learning "
              "these operations first makes the layers of a CNN, used in AAT 2, much easier to understand.")
    bullets(doc, ["Element-wise operation: Tensor a, Tensor b → same position combined → Tensor of the same shape",
                  "Matrix multiplication: rows of a × columns of b → dot products → new matrix",
                  "Reduction: Tensor → sum, mean or maximum along an axis → smaller tensor or a single value"])

    heading(doc, "2. Objectives")
    para(doc, "The aim is to create tensors in TensorFlow and perform the basic operations on which neural networks are "
              "built. The experiment has five objectives:")
    bullets(doc, ["Create scalar, vector, matrix and 3D tensors, and special tensors (zeros, ones and random values).",
                  "Perform element-wise arithmetic: addition, subtraction, multiplication and division.",
                  "Perform matrix operations: matrix multiplication, transpose, reshape and element-wise square.",
                  "Apply reduction operations: sum, mean, maximum and argmax.",
                  "Compare immutable constants with mutable variables, which hold the weights of a model."], numbered=True)
    para(doc, "Expected outcomes", size=12.5, align="left", space_after=3)
    bullets(doc, ["A working notebook in which every operation prints its result, shape and data type.",
                  "A clear difference between element-wise multiplication and matrix multiplication.",
                  "An understanding of how shape and data type change after each operation.",
                  "A link between these operations and the layers of a neural network."], numbered=True)

    heading(doc, "3. Case Description / Background Information")
    para(doc, "The case uses two small 2×2 integer matrices, a = [[1, 2], [3, 4]] and b = [[5, 6], [7, 8]]. Small "
              "values were chosen on purpose so that every output can be verified by hand. The same operations work "
              "unchanged on tensors with millions of elements.")
    para(doc, "Experiment overview", size=12.5, align="left", space_after=3)
    table(doc, [["Attribute", "Value"], ["Framework", "TensorFlow 2.20.0 (Python 3)"],
                ["Input tensors", "a = [[1, 2], [3, 4]], b = [[5, 6], [7, 8]]"], ["Shape of a and b", "(2, 2)"],
                ["Data type", "int32 (float64 after division)"], ["Operations applied", "11, plus tensor creation"],
                ["Environment", "Jupyter notebook"]], widths=[2.2, 3.6])
    para(doc, "Key concepts", size=12.5, align="left", space_after=3)
    bullets(doc, ["**Rank:** the number of dimensions of a tensor. A scalar has rank 0, a vector rank 1, a matrix rank 2.",
                  "**Shape:** the number of elements along each dimension, for example (2, 2) for a 2×2 matrix.",
                  "**Data type (dtype):** the type of every element, such as int32 or float32. All elements share one dtype.",
                  "**Element-wise operation:** combines the elements in the same position of two tensors of the same shape.",
                  "**Reduction:** collapses one or more dimensions into a single summary value."])
    figno = fig(doc, F + "aat1_tensor_ranks.png", figno, "Tensors of rank 0 to 3 used in the experiment", 6.0)
    para(doc, "Operations used", size=12.5, align="left", space_after=3)
    table(doc, [["Group", "TensorFlow function", "What it does"],
                ["Creation", "tf.constant, tf.zeros, tf.ones, tf.random.uniform", "Builds a tensor from values, or filled with 0, 1 or random numbers"],
                ["Element-wise", "tf.add, tf.subtract, tf.multiply, tf.divide", "Combines elements in the same position"],
                ["Matrix", "tf.matmul", "Row by column dot product, as in a dense layer"],
                ["Shape", "tf.transpose, tf.reshape", "Swaps rows and columns, or gives the data a new shape"],
                ["Power", "tf.square", "Squares every element"],
                ["Reduction", "tf.reduce_sum, tf.reduce_mean, tf.reduce_max, tf.argmax", "Summarises a tensor or finds the position of the maximum"],
                ["State", "tf.constant, tf.Variable", "Fixed value versus a value that can be updated"]],
          widths=[1.1, 2.6, 2.6], align=["center", "left", "left"])

    heading(doc, "4. Analysis and Discussion")
    para(doc, "The notebook follows a six-stage workflow. Each stage builds on the previous one, starting with tensor "
              "creation and ending with variables, which are the trainable part of every model.")
    figno = fig(doc, F + "aat1_workflow.png", figno, "Workflow of the experiment", 6.4)
    para(doc, "Implementation", size=12.5, align="left", space_after=3)
    bullets(doc, ["Import TensorFlow and print its version (2.20.0).",
                  "Create a scalar, a vector, a matrix and a 3D tensor with tf.constant, and check their shapes.",
                  "Create special tensors with tf.zeros, tf.ones and tf.random.uniform.",
                  "Define the two matrices a and b.",
                  "Apply tf.add, tf.subtract, tf.multiply and tf.divide.",
                  "Apply tf.matmul, tf.transpose, tf.reshape and tf.square.",
                  "Apply tf.reduce_sum, tf.reduce_mean, tf.reduce_max and tf.argmax.",
                  "Create a constant and a variable, and update the variable with assign()."], numbered=True)
    code_block(doc, """import tensorflow as tf
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

tf.add(a, b)            # element-wise sum
tf.multiply(a, b)       # element-wise product
tf.matmul(a, b)         # matrix product (rows x columns)
tf.reshape(a, [4, 1])   # same 4 values, new shape
tf.reduce_sum(a)        # 10""")
    para(doc, "Element-wise multiplication versus matrix multiplication", size=12.5, align="left", space_after=3)
    para(doc, "The two kinds of multiplication are the most common source of confusion. tf.multiply pairs each element "
              "with the element in the same position, so both tensors must have the same shape. tf.matmul takes the dot "
              "product of a row of the first matrix with a column of the second, so the number of columns of a must "
              "equal the number of rows of b. A dense neural network layer computes tf.matmul(x, W) + b.")
    figno = fig(doc, F + "aat1_mul_vs_matmul.png", figno, "Element-wise multiplication compared with matrix multiplication", 6.3)
    table(doc, [["tf.multiply (element-wise)", "tf.matmul (matrix product)"],
                ["Shapes must match (or broadcast)", "Inner dimensions must match"],
                ["1×5, 2×6, 3×7, 4×8", "1×5 + 2×7 = 19, and so on"],
                ["Result [[5, 12], [21, 32]]", "Result [[19, 22], [43, 50]]"],
                ["Commutative: a·b = b·a", "Not commutative: ab ≠ ba in general"],
                ["Used for masks and scaling", "Used in dense and attention layers"]], widths=[3.0, 3.0])
    para(doc, "Shape, data type and reduction", size=12.5, align="left", space_after=3)
    para(doc, "Shape changes do not change the data. tf.transpose swaps rows and columns, and tf.reshape lays the same "
              "four values out as a 4×1 column, which is only allowed because 2×2 and 4×1 both hold four "
              "elements. Data types do change in one place: dividing two int32 tensors returns float64, because the "
              "result of a division is not always a whole number. Reductions remove dimensions. Note that tf.reduce_mean "
              "on an int32 tensor returns 2 instead of 2.5, because integer arithmetic drops the fraction, so the tensor "
              "should be cast to float before taking a mean.")
    figno = fig(doc, F + "aat1_shape_reduce.png", figno, "Transpose, reshape and reduction operations on tensor a", 6.3)
    para(doc, "Constants versus variables", size=12.5, align="left", space_after=3)
    table(doc, [["tf.constant", "tf.Variable"], ["Value is fixed after creation", "Value can be changed with assign()"],
                ["Used for input data and fixed values", "Used for weights and biases"],
                ["Calling assign() raises an error", "assign(20) changed the value from 10 to 20"],
                ["Not tracked for training", "Tracked by the optimiser during training"]], widths=[3.0, 3.0])
    para(doc, "Strengths and limitations of the approach", size=12.5, align="left", space_after=3)
    table(doc, [["Strengths", "Limitations"],
                ["Small inputs make every result easy to verify by hand", "2×2 tensors do not show the speed benefit of a GPU"],
                ["Covers creation, arithmetic, shape, reduction and state", "Broadcasting between different shapes is not tested"],
                ["Each cell prints shape and dtype as well as values", "Integer tensors hide fractions in mean and division"]],
          widths=[3.15, 3.15], align=["left", "left"])

    heading(doc, "5. Results")
    para(doc, "All operations ran without errors and every output matched the value calculated by hand. The screenshots "
              "below are taken from the executed notebook.")
    for a, b, cap in [("1a", "1b", "Creating tensors, special tensors and the input matrices a and b"),
                      ("2a", "2b", "Element-wise operations, matrix multiplication and division"),
                      ("3a", "3b", "Transpose, reshape, square, reductions, and constants versus variables")]:
        figure_pair(doc, F + f"aat1_shot_{a}.png", F + f"aat1_shot_{b}.png", f"Fig {figno}: {cap}")
        figno += 1
    para(doc, "Summary of results", size=12.5, align="left", space_after=3)
    table(doc, [["Operation", "Result", "Shape", "Dtype"],
                ["tf.add(a, b)", "[[6, 8], [10, 12]]", "(2, 2)", "int32"],
                ["tf.subtract(a, b)", "[[-4, -4], [-4, -4]]", "(2, 2)", "int32"],
                ["tf.multiply(a, b)", "[[5, 12], [21, 32]]", "(2, 2)", "int32"],
                ["tf.matmul(a, b)", "[[19, 22], [43, 50]]", "(2, 2)", "int32"],
                ["tf.divide(a, b)", "[[0.2, 0.333], [0.429, 0.5]]", "(2, 2)", "float64"],
                ["tf.transpose(a)", "[[1, 3], [2, 4]]", "(2, 2)", "int32"],
                ["tf.reshape(a, [4, 1])", "[[1], [2], [3], [4]]", "(4, 1)", "int32"],
                ["tf.square(a)", "[[1, 4], [9, 16]]", "(2, 2)", "int32"],
                ["tf.reduce_sum(a)", "10", "()", "int32"],
                ["tf.reduce_mean(a)", "2", "()", "int32"],
                ["tf.reduce_max(a)", "4", "()", "int32"],
                ["tf.argmax(a, axis=1)", "[1, 1]", "(2,)", "int64"]], widths=[1.9, 2.3, 0.9, 0.9], size=10)
    para(doc, "Discussion of the results", size=12.5, align="left", space_after=3)
    bullets(doc, ["**Element-wise results are correct:** each output cell depends only on the two input cells in the same position.",
                  "**Matrix product differs from element-wise product:** [[19, 22], [43, 50]] against [[5, 12], [21, 32]], which confirms that the two operations must not be mixed up.",
                  "**Division changes the data type:** int32 inputs give a float64 output.",
                  "**Integer mean is truncated:** the true mean of a is 2.5, but the int32 result is 2.",
                  "**Variables are mutable:** assign(20) changed the variable, while the constant stayed at 10."])

    heading(doc, "6. Conclusion")
    para(doc, "TensorFlow tensors of rank 0 to 3 were created, and eleven basic operations were applied and verified by "
              "hand. The project achieved the following:")
    bullets(doc, ["Created scalar, vector, matrix, 3D and special tensors and checked their shapes and data types.",
                  "Performed element-wise addition, subtraction, multiplication and division.",
                  "Showed that tf.matmul and tf.multiply give different results on the same inputs.",
                  "Applied transpose, reshape, square and the sum, mean, max and argmax reductions.",
                  "Showed that tf.Variable can be updated while tf.constant cannot."])
    para(doc, "These operations are the building blocks of every layer in a neural network: a dense layer is a matrix "
              "multiplication followed by an addition, a convolution is a sliding element-wise multiplication followed by "
              "a sum, and training updates tf.Variable weights. The CNN in AAT 2 is built entirely from these operations. "
              "Future work could cover broadcasting, operations on GPU with large tensors, and automatic differentiation "
              "with tf.GradientTape.")
    return figno


def fig(doc, path, n, caption, width):
    figure(doc, path, f"Fig {n}: {caption}", width)
    return n + 1
