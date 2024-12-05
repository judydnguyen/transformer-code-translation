public static long Factorial(int n) => n <= 1 ? 1 : n * Factorial(n - 1);
public static long Factorial(int n) => Enumerable.Range(1, n).Aggregate(1L, (acc, x) => acc * x);
public class HelloWorld { public static void Main(string[] args) { Console.WriteLine("Hello, World!"); } }
public class Calculator { public static int Add(int a, int b) { return a + b; } }
public class LoopExample { public static void PrintNumbers(int n) { for (int i = 0; i < n; i++) { Console.WriteLine(i); } } }
public class ConditionExample { public static string GetDay(int day) { switch (day) { case 1: return "Monday"; case 2: return "Tuesday"; default: return "Invalid"; } } }
public class ExceptionExample { public static void Divide(int a, int b) { try { Console.WriteLine(a / b); } catch (DivideByZeroException e) { Console.WriteLine("Cannot divide by zero"); } } }