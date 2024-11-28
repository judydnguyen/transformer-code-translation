public class HelloWorld { public static void main(String[] args) { System.out.println("Hello, World!"); } }
public static long factorial(int n) { if (n <= 1) { return 1; } return n * factorial(n - 1); }
public static long factorial(int n) { long result = 1; for (int i = 2; i <= n; i++) { result *= i; } return result; }
public static int add(int a, int b) { return a + b; }
public static int subtract(int a, int b) { return a - b; }
public static void printNumbers(int n) { for (int i = 0; i < n; i++) { System.out.println(i); } }
public static boolean isEven(int n) { return n % 2 == 0; }
public static String getDay(int day) { switch (day) { case 1: return "Monday"; case 2: return "Tuesday"; default: return "Invalid"; } }
public static void divide(int a, int b) { try { System.out.println(a / b); } catch (ArithmeticException e) { System.out.println("Cannot divide by zero"); } }
public static void printEvenNumbers(int n) { for (int i = 0; i <= n; i++) { if (i % 2 == 0) { System.out.println(i); } } }
public static void checkNumber(int x) { if (x > 0) { if (x < 10) { System.out.println("Between 1 and 9"); } else { System.out.println("10 or more"); } } else { System.out.println("Non-positive"); } }
public static String reverseString(String input) { return new StringBuilder(input).reverse().toString(); }
public static boolean isPalindrome(String input) { String reversed = new StringBuilder(input).reverse().toString(); return input.equals(reversed); }
public static int sumArray(int[] arr) { int sum = 0; for (int num : arr) { sum += num; } return sum; }
public static int findIndex(int[] arr, int target) { for (int i = 0; i < arr.length; i++) { if (arr[i] == target) { return i; } } return -1; }
public static void bubbleSort(int[] arr) { for (int i = 0; i < arr.length - 1; i++) { for (int j = 0; j < arr.length - i - 1; j++) { if (arr[j] > arr[j + 1]) { int temp = arr[j]; arr[j] = arr[j + 1]; arr[j + 1] = temp; } } } }
public static boolean isPrime(int n) { if (n < 2) return false; for (int i = 2; i <= Math.sqrt(n); i++) { if (n % i == 0) return false; } return true; }
public static int[] fibonacci(int n) { int[] fib = new int[n]; fib[0] = 0; if (n > 1) fib[1] = 1; for (int i = 2; i < n; i++) { fib[i] = fib[i - 1] + fib[i - 2]; } return fib; }
public static int[][] multiplyMatrices(int[][] a, int[][] b) { int rows = a.length; int cols = b[0].length; int[][] result = new int[rows][cols]; for (int i = 0; i < rows; i++) { for (int j = 0; j < cols; j++) { for (int k = 0; k < a[0].length; k++) { result[i][j] += a[i][k] * b[k][j]; } } } return result; }
public static String readFile(String path) { try { return new String(Files.readAllBytes(Paths.get(path))); } catch (IOException e) { return ""; } }