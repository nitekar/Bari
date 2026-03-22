/**
 * supabaseStorage.ts — Conjunctiva image upload / retrieval
 *
 * Images are stored in the "conjunctiva-images" bucket under a path
 * scoped to the authenticated user:  <user_id>/<timestamp>.jpg
 *
 * RLS policies on the bucket ensure users can only access their own images.
 */
import { supabase } from './supabase';

const BUCKET = 'conjunctiva-images';

// ── Upload ───────────────────────────────────────────────────────────────────

/**
 * Upload a local image to Supabase Storage.
 * @returns The storage path (e.g. "<user_id>/1711052400000.jpg")
 */
export async function uploadConjunctivaImage(
  userId: string,
  imageUri: string,
): Promise<string> {
  // Read the file as a blob
  const response = await fetch(imageUri);
  const blob = await response.blob();

  const filePath = `${userId}/${Date.now()}.jpg`;

  const { error } = await supabase.storage
    .from(BUCKET)
    .upload(filePath, blob, {
      contentType: 'image/jpeg',
      upsert: false,
    });

  if (error) throw new Error(`Image upload failed: ${error.message}`);

  return filePath;
}

// ── Retrieve ─────────────────────────────────────────────────────────────────

/**
 * Get a signed URL for a stored image (valid for 1 hour).
 */
export async function getImageUrl(
  filePath: string,
): Promise<string> {
  const { data, error } = await supabase.storage
    .from(BUCKET)
    .createSignedUrl(filePath, 60 * 60); // 1 hour

  if (error) throw new Error(`Failed to get image URL: ${error.message}`);
  return data.signedUrl;
}
