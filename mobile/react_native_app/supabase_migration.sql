-- ============================================================================
-- Supabase Migration for Anemia Screening App
-- Run this in the Supabase SQL Editor (https://app.supabase.com → SQL Editor)
-- ============================================================================

-- ── Screenings table ─────────────────────────────────────────────────────────

create table if not exists screenings (
  id               uuid default gen_random_uuid() primary key,
  user_id          uuid references auth.users(id) on delete cascade not null,
  prediction       text not null,
  confidence       real not null,
  mode             text not null check (mode in ('tabular', 'image', 'multimodal')),
  age              integer,
  gender           integer,
  hb_level         real,
  image_url        text,
  patient_name     text,
  patient_location text,
  created_at       timestamptz default now()
);

-- Add columns if table already exists (safe to run on existing DB)
alter table screenings add column if not exists patient_name     text;
alter table screenings add column if not exists patient_location text;

-- ── Analytics events table ───────────────────────────────────────────────────

create table if not exists analytics_events (
  id          uuid default gen_random_uuid() primary key,
  user_id     uuid references auth.users(id) on delete cascade not null,
  event_name  text not null,
  metadata    jsonb default '{}'::jsonb,
  created_at  timestamptz default now()
);

-- ── Row-Level Security ───────────────────────────────────────────────────────

alter table screenings enable row level security;
alter table analytics_events enable row level security;

-- Screenings policies
create policy "Users see own screenings"
  on screenings for select
  using (auth.uid() = user_id);

create policy "Users insert own screenings"
  on screenings for insert
  with check (auth.uid() = user_id);

-- Analytics events policies
create policy "Users see own events"
  on analytics_events for select
  using (auth.uid() = user_id);

create policy "Users insert own events"
  on analytics_events for insert
  with check (auth.uid() = user_id);

-- ── Storage bucket ───────────────────────────────────────────────────────────

insert into storage.buckets (id, name, public)
  values ('conjunctiva-images', 'conjunctiva-images', false)
  on conflict (id) do nothing;

-- Storage policies (images scoped by user_id folder)
create policy "Users upload own images"
  on storage.objects for insert
  with check (
    bucket_id = 'conjunctiva-images'
    and auth.uid()::text = (storage.foldername(name))[1]
  );

create policy "Users view own images"
  on storage.objects for select
  using (
    bucket_id = 'conjunctiva-images'
    and auth.uid()::text = (storage.foldername(name))[1]
  );
