import { NextResponse } from 'next/server'

export const maxDuration = 120

export async function POST(request: Request) {
  const body = await request.json()
  const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'

  try {
    const response = await fetch(`${backendUrl}/api/chat/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: body.user_id || 'demo_user',
        message: body.message,
        history: body.history || []
      })
    })

    if (!response.ok) {
      return NextResponse.json({
        reply: "backend hiccup — check if the server is running",
        tool_calls: [],
        tailor_flow: false
      })
    }

    const data = await response.json()
    return NextResponse.json({
      reply: data.reply || '',
      tool_calls: data.tool_calls || [],
      tailor_flow: data.tailor_flow ?? false,
      model: data.model || ''
    })

  } catch {
    return NextResponse.json({
      reply: "can't reach backend — is it running on :8000?",
      tool_calls: [],
      tailor_flow: false
    })
  }
}
