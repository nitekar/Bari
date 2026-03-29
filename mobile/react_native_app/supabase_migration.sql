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

-- ── Profiles table (role-based) ───────────────────────────────────────────────

create table if not exists profiles (
  id          uuid references auth.users(id) on delete cascade primary key,
  role        text not null default 'parent' check (role in ('admin', 'chw', 'parent')),
  full_name   text,
  created_at  timestamptz default now(),
  updated_at  timestamptz default now()
);

alter table profiles enable row level security;

-- Auto-create profile on sign-up using user metadata
-- SECURITY: Rejects 'admin' role from self-registration.
-- Admin accounts must be created manually in the Supabase dashboard.
create or replace function public.handle_new_user()
returns trigger as $$
declare
  claimed_role text;
begin
  claimed_role := coalesce(new.raw_user_meta_data->>'role', 'parent');
  -- Only allow 'chw' and 'parent' via self-signup
  if claimed_role not in ('chw', 'parent') then
    claimed_role := 'parent';
  end if;

  insert into public.profiles (id, role, full_name)
  values (
    new.id,
    claimed_role,
    coalesce(new.raw_user_meta_data->>'full_name', '')
  )
  on conflict (id) do nothing;
  return new;
end;
$$ language plpgsql security definer;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute procedure public.handle_new_user();

-- RLS policies for profiles
create policy "Users can view own profile"
  on profiles for select
  using (auth.uid() = id);

create policy "Users can update own profile"
  on profiles for update
  using (auth.uid() = id);

-- Admins can view all profiles
create policy "Admins view all profiles"
  on profiles for select
  using (
    exists (
      select 1 from profiles p
      where p.id = auth.uid() and p.role = 'admin'
    )
  );

-- ── Children table ────────────────────────────────────────────────────────────

create table if not exists children (
  id          uuid default gen_random_uuid() primary key,
  parent_id   uuid references auth.users(id) on delete cascade not null,
  name        text,
  age_months  integer,
  gender      text check (gender in ('male', 'female')),
  created_at  timestamptz default now(),
  updated_at  timestamptz default now()
);

alter table children enable row level security;

create policy "Parents see own children"
  on children for select
  using (auth.uid() = parent_id);

create policy "Parents insert own children"
  on children for insert
  with check (auth.uid() = parent_id);

create policy "Parents update own children"
  on children for update
  using (auth.uid() = parent_id);

-- ── Sleep logs table ──────────────────────────────────────────────────────────

create table if not exists sleep_logs (
  id              uuid default gen_random_uuid() primary key,
  user_id         uuid references auth.users(id) on delete cascade not null,
  start_time      text not null,
  end_time        text not null,
  duration_hours  real not null,
  notes           text default '',
  date            date not null default current_date,
  created_at      timestamptz default now()
);

alter table sleep_logs enable row level security;

create policy "Users see own sleep logs"
  on sleep_logs for select
  using (auth.uid() = user_id);

create policy "Users insert own sleep logs"
  on sleep_logs for insert
  with check (auth.uid() = user_id);

-- ── Feeding logs table ────────────────────────────────────────────────────────

create table if not exists feeding_logs (
  id           uuid default gen_random_uuid() primary key,
  user_id      uuid references auth.users(id) on delete cascade not null,
  type         text not null check (type in ('breastfeeding', 'formula', 'solid')),
  time         text not null,
  quantity_ml  real,
  notes        text default '',
  date         date not null default current_date,
  created_at   timestamptz default now()
);

alter table feeding_logs enable row level security;

create policy "Users see own feeding logs"
  on feeding_logs for select
  using (auth.uid() = user_id);

create policy "Users insert own feeding logs"
  on feeding_logs for insert
  with check (auth.uid() = user_id);
