import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

// ─── Generate a demo JWT for local dev ────────────────────────────────────────
// Uses the same JWT_SECRET_KEY as the FastAPI backend.
// This runs server-side only — secret never reaches the browser.
async function makeDemoToken(userId: string): Promise<string> {
  const secret = process.env.JWT_SECRET_KEY ?? "demo_secret";
  const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" })).replace(
    /=/g,
    ""
  );
  const now = Math.floor(Date.now() / 1000);
  const payload = btoa(
    JSON.stringify({ sub: userId, user_id: userId, exp: now + 86400 })
  ).replace(/=/g, "");

  // HMAC-SHA256 via Web Crypto (available in Next.js Edge/Node runtime)
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign(
    "HMAC",
    key,
    enc.encode(`${header}.${payload}`)
  );
  const sigB64 = btoa(String.fromCharCode(...new Uint8Array(sig)))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=/g, "");

  return `${header}.${payload}.${sigB64}`;
}

export async function POST(request: NextRequest) {
  // Prefer the real httpOnly cookie (set by backend login flow)
  const cookieStore = await cookies();
  let token = cookieStore.get("shaaru_token")?.value;

  // For local demo: generate a signed JWT if no cookie is set
  if (!token) {
    if (process.env.NODE_ENV === "production") {
      return NextResponse.json(
        { error: "Unauthorised — no session token" },
        { status: 401 }
      );
    }
    // dev/demo: mint a short-lived token for demo_user
    token = await makeDemoToken("demo_user");
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${BACKEND_URL}/api/tailor/reference`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify(body),
    });

    const data = await upstream.json();

    if (!upstream.ok) {
      return NextResponse.json(data, { status: upstream.status });
    }

    return NextResponse.json(data);
  } catch (err) {
    console.error("[tailor/reference proxy] upstream error:", err);
    return NextResponse.json(
      { error: "Upstream service unavailable" },
      { status: 502 }
    );
  }
}
