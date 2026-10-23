import report from "../../doc.tex?raw";

export function GET() {
  return new Response(report, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
      "Content-Disposition": 'attachment; filename="doc.tex"',
    },
  });
}
