import readme from "../../README.md?raw";

export function GET() {
  return new Response(readme, {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
}
