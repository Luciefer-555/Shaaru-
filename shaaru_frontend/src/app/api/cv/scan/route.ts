import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 120; // tell Next.js to wait 2 min

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  try {
    const upstream = await fetch(`${BACKEND_URL}/api/cv/scan`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    const data = await upstream.json();

    if (!upstream.ok) {
      return NextResponse.json(data, { status: upstream.status });
    }

    return NextResponse.json(data);
  } catch (err) {
    console.error("[cv/scan proxy] upstream error:", err);
    return NextResponse.json(
      { error: "scan failed", items: [], guidance: "try again" },
      { status: 500 }
    );
  }
}
