import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // `standalone` emits a self-contained server bundle under .next/standalone
  // including only the node_modules actually traced from the entry points.
  // The Docker runtime stage copies that directory and runs `node server.js`
  // without needing the full repo or `pnpm install`.
  output: "standalone",

  images: {
    // The FastAPI backend serves uploaded property images. Allow next/image
    // to optimise them in both the container network (api:8000) and on the
    // developer host (localhost:8000).
    remotePatterns: [
      { protocol: "http", hostname: "api", port: "8000", pathname: "/api/v1/images/**" },
      { protocol: "http", hostname: "localhost", port: "8000", pathname: "/api/v1/images/**" },
    ],
  },
};

export default nextConfig;
