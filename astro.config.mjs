// @ts-check
import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import react from "@astrojs/react";

import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const baseSegments = (process.env.ASTRO_BASE ?? "").split("/").filter(Boolean);
const base = baseSegments.length ? "/" + baseSegments.join("/") + "/" : "/";

export default defineConfig({
  output: "static",
  base,
  integrations: [mdx(), react()],

  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },

});
