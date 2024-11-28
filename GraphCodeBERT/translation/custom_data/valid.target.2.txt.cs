public class HelloWorld { public static void Main(string[] args) { Console.WriteLine("Hello, World!"); } }
public static long Factorial(int n) { if (n <= 1) { return 1; } return n * Factorial(n - 1); }
public static long Factorial(int n) { long result = 1; for (int i = 2; i <= n; i++) { result *= i; } return result; }
public static int Add(int a, int b) { return a + b; }
public static int Subtract(int a, int b) { return a - b; }
public static void PrintNumbers(int n) { for (int i = 0; i < n; i++) { Console.WriteLine(i); } }
public static bool IsEven(int n) { return n % 2 == 0; }
public static string GetDay(int day) { switch (day) { case 1: return "Monday"; case 2: return "Tuesday"; default: return "Invalid"; } }
public static void Divide(int a, int b) { try { Console.WriteLine(a / b); } catch (DivideByZeroException) { Console.WriteLine("Cannot divide by zero"); } }
public static void PrintEvenNumbers(int n) { for (int i = 0; i <= n; i++) { if (i % 2 == 0) { Console.WriteLine(i); } } }
public static void CheckNumber(int x) { if (x > 0) { if (x < 10) { Console.WriteLine("Between 1 and 9"); } else { Console.WriteLine("10 or more"); } } else { Console.WriteLine("Non-positive"); } }
public static string ReverseString(string input) { return new string(input.Reverse().ToArray()); }
public static bool IsPalindrome(string input) { string reversed = new string(input.Reverse().ToArray()); return input == reversed; }
public static int SumArray(int[] arr) { return arr.Sum(); }
public static int FindIndex(int[] arr, int target) { for (int i = 0; i < arr.Length; i++) { if (arr[i] == target) { return i; } } return -1; }
public static void BubbleSort(int[] arr) { for (int i = 0; i < arr.Length - 1; i++) { for (int j = 0; j < arr.Length - i - 1; j++) { if (arr[j] > arr[j + 1]) { int temp = arr[j]; arr[j] = arr[j + 1]; arr[j + 1] = temp; } } } }
public static bool IsPrime(int n) { if (n < 2) return false; for (int i = 2; i <= Math.Sqrt(n); i++) { if (n % i == 0) return false; } return true; }
public static int[] Fibonacci(int n) { int[] fib = new int[n]; fib[0] = 0; if (n > 1) fib[1] = 1; for (int i = 2; i < n; i++) { fib[i] = fib[i - 1] + fib[i - 2]; } return fib; }
public static int[,] MultiplyMatrices(int[,] a, int[,] b) { int rows = a.GetLength(0); int cols = b.GetLength(1); int[,] result = new int[rows, cols]; for (int i = 0; i < rows; i++) { for (int j = 0; j < cols; j++) { for (int k = 0; k < a.GetLength(1); k++) { result[i, j] += a[i, k] * b[k, j]; } } } return result; }
public static string ReadFile(string path) { return File.ReadAllText(path); }