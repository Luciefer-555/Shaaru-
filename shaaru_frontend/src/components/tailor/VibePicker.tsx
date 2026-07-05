'use client'
import { useState } from 'react'

const AESTHETICS = [
  { id: 'y2k', label: 'Y2K Street Ethnic', female: '/aesthetics/Y2k_F.jpg', male: '/aesthetics/Y2K_M.jpg' },
  { id: 'office_core', label: 'Office Core', female: '/aesthetics/Office_core_F.jpg', male: '/aesthetics/office_core_M.jpg' },
  { id: 'dark_academia', label: 'Dark Academia', female: '/aesthetics/DA_F.jpg', male: '/aesthetics/DA_M.jpg' },
  { id: 'boho_fusion', label: 'Boho Fusion', female: '/aesthetics/BOHO_FUSION_F.jpg', male: '/aesthetics/BOHO_FUSION_M.jpg' },
  { id: 'indian_max', label: 'Indian Maximalist', female: '/aesthetics/Indian_Max_F.jpg', male: '/aesthetics/Indian_Max_M.jpg' },
  { id: 'indian_min', label: 'Indian Minimal', female: '/aesthetics/INDIAN_MIN_F.jpg', male: '/aesthetics/Indian_MIni_M.jpg' },
]

interface VibePickerProps {
  pronouns: string
  userId: string
  onComplete: (selected: string[]) => void
}

export default function VibePicker({ pronouns, userId, onComplete }: VibePickerProps) {
  const [selected, setSelected] = useState<string[]>([])
  const [submitting, setSubmitting] = useState(false)

  const getImage = (a: typeof AESTHETICS[0]) => {
    if (pronouns === 'he/him') return a.male
    return a.female
  }

  const toggle = (id: string) => {
    setSelected(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id])
  }

  const handleSubmit = async () => {
    if (selected.length === 0) return
    setSubmitting(true)
    try {
      await fetch('/api/onboarding/aesthetics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, aesthetics: selected })
      })
      onComplete(selected)
    } catch (e) {
      console.error(e)
    } finally {
      setSubmitting(false)
    }
  }

  const isTheyThem = pronouns === 'they/them'

  return (
    <div style={{ width: '100%', marginTop: '8px' }}>
      <p style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '10px' }}>
        tap everything that feels like you
      </p>

      <div style={{
        display: 'flex',
        overflowX: 'auto',
        gap: '10px',
        paddingBottom: '8px',
        scrollSnapType: 'x mandatory',
        WebkitOverflowScrolling: 'touch',
        scrollbarWidth: 'none',
      }}>
        {AESTHETICS.map(aesthetic => {
          const isSelected = selected.includes(aesthetic.id)

          if (isTheyThem) {
            return (
              <div
                key={aesthetic.id}
                onClick={() => toggle(aesthetic.id)}
                style={{
                  flexShrink: 0,
                  width: '160px',
                  scrollSnapAlign: 'start',
                  cursor: 'pointer',
                  position: 'relative',
                  borderRadius: '10px',
                  overflow: 'hidden',
                  border: isSelected ? '2px solid #e63946' : '2px solid transparent',
                }}
              >
                <div style={{ display: 'flex', height: '220px' }}>
                  <img src={aesthetic.female} alt="" style={{ width: '50%', height: '100%', objectFit: 'cover' }} />
                  <img src={aesthetic.male} alt="" style={{ width: '50%', height: '100%', objectFit: 'cover' }} />
                </div>
                <div style={{
                  position: 'absolute', bottom: 0, left: 0, right: 0,
                  background: 'linear-gradient(transparent, rgba(0,0,0,0.75))',
                  padding: '20px 8px 8px',
                  color: 'white', fontSize: '11px', fontWeight: 500
                }}>
                  {aesthetic.label}
                </div>
                {isSelected && (
                  <div style={{
                    position: 'absolute', top: '6px', right: '6px',
                    background: '#e63946', color: 'white',
                    width: '20px', height: '20px', borderRadius: '50%',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '12px', fontWeight: 700
                  }}>✓</div>
                )}
              </div>
            )
          }

          return (
            <div
              key={aesthetic.id}
              onClick={() => toggle(aesthetic.id)}
              style={{
                flexShrink: 0,
                width: '140px',
                height: '220px',
                scrollSnapAlign: 'start',
                cursor: 'pointer',
                position: 'relative',
                borderRadius: '10px',
                overflow: 'hidden',
                border: isSelected ? '2px solid #e63946' : '2px solid transparent',
                opacity: isSelected ? 1 : 0.8,
                transition: 'all 0.15s'
              }}
            >
              <img
                src={getImage(aesthetic)}
                alt={aesthetic.label}
                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              />
              <div style={{
                position: 'absolute', bottom: 0, left: 0, right: 0,
                background: 'linear-gradient(transparent, rgba(0,0,0,0.75))',
                padding: '20px 8px 8px',
                color: 'white', fontSize: '11px', fontWeight: 500
              }}>
                {aesthetic.label}
              </div>
              {isSelected && (
                <div style={{
                  position: 'absolute', top: '6px', right: '6px',
                  background: '#e63946', color: 'white',
                  width: '20px', height: '20px', borderRadius: '50%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '12px', fontWeight: 700
                }}>✓</div>
              )}
            </div>
          )
        })}
      </div>

      <button
        onClick={handleSubmit}
        disabled={selected.length === 0 || submitting}
        style={{
          marginTop: '10px',
          width: '100%',
          padding: '11px',
          background: selected.length > 0 ? '#e63946' : 'var(--surface-1)',
          color: selected.length > 0 ? 'white' : 'var(--text-muted)',
          border: 'none',
          borderRadius: '10px',
          fontSize: '13px',
          fontWeight: 500,
          cursor: selected.length > 0 ? 'pointer' : 'not-allowed',
          transition: 'all 0.15s'
        }}
      >
        {submitting ? 'saving...' : `this is me${selected.length > 0 ? ` (${selected.length} picked)` : ''} →`}
      </button>
    </div>
  )
}
