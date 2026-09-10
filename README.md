# SHAARU — AI Fashion Styling Platform for India

SHAARU is a two-sided AI fashion platform: an AI stylist for everyday users (wardrobe understanding, outfit planning, style profiling) and an AI atelier for designers (design generation, video simulation). This repo covers the core system — computer vision pipeline, styling knowledge graph, and backend.

Built solo, from architecture to deployment, as the flagship product in a longer founder roadmap.

## Problem

Most styling apps rely on long onboarding questionnaires or generic "trend" recommendations that ignore what a person already owns and how they actually dress. SHAARU instead reads a user's existing wardrobe directly from photos and grounds every recommendation in an India-specific fashion knowledge graph — no face or skin photo uploads required.

## System Architecture

```
Photo → RF-DETR pre-detection (bbox priors, ~220-230ms CPU)
      → VLM enrichment (NVIDIA NIM — Llama-3.2-11B fast pass, 90B async crop-based fabric detail)
      → IoU-based bbox merge (VLM labels retained, RF-DETR boxes adopted)
      → Structured garment record (type, color, pattern, fabric)
      → Neo4j styling knowledge graph (167 fabric nodes, 57 Indian brands across 7 categories)
      → Outfit combo generation
```

Backend: FastAPI on Railway. Frontend: Next.js 15 on Vercel. Data layer: Neo4j (styling graph) + MongoDB (structured records, cache).

## Computer Vision Pipeline

The core engineering challenge is accurate garment detection and classification from casual wardrobe photos — not studio product shots.

**Approach:** a lightweight local RF-DETR model runs first as a detection pre-pass, producing bounding-box priors that get injected into the VLM prompt rather than triggering an additional model call. This keeps VLM call count fixed at 2 per scan while meaningfully improving box accuracy on real-world photos.

**Measured results** (21-image held-out test set, bbox-cited grading):
- Type classification: 78.9%
- Color classification: 71.1%
- Pattern classification: 71.1%
- Fallback rate: 0%
- End-to-end enrichment latency: ~10.6s

**Reliability work:** the pipeline runs on a bounded async retry wrapper (hard timeout, jittered backoff, model fallback) rather than a raw client call, and JWT auth is enforced on all wardrobe/chat/profile endpoints.

Known limitation, tracked openly: RF-DETR trouser/pants detection is a genuine training-data gap (confidence scores as low as 0.0–0.38 on garments in plain view) — estimated 500-1000 additional annotated images needed to close it via retrain. Documenting a known gap honestly is more useful to a reader than hiding it.

## Knowledge Base

- Fabric taxonomy: 167 nodes in Neo4j, rebuilt from a flat 150+ term list into a hierarchical two-tier system
- Brand knowledge base: 57 Indian fashion brands across 7 categories (luxury ethnic, streetwear, couture, footwear, accessories, bags, contemporary)
- Four-layer fallback for unknown items: Neo4j cache → on-demand extraction → real-time search with write-back → general fashion knowledge

## Tech Stack

`Python` `FastAPI` `Next.js 15` `NVIDIA NIM (Llama-3.2 VLM family)` `RF-DETR` `Neo4j` `MongoDB` `Railway` `Vercel`

## Status

Backend is fully verified end-to-end (30+ message test suite passing). Active work is closing the CV accuracy gap (pants/trouser detection) before moving to GPU-hosted inference infrastructure.

---

*Built and maintained solo. Architecture decisions, prompt engineering, and CV pipeline design are original work; NVIDIA NIM is used as the hosted inference layer.*
