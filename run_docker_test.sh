echo "Fetching untagged images..."
# Filter images with empty tags
UNTAGGED_IMAGES=$(gcloud artifacts docker images list \
    europe-west4-docker.pkg.dev/mlops-448109/mlops-artifact-registry \
    --format="json" | jq -r '.[] | select(.tags == "") | .metadata.name')

if [ -z "$UNTAGGED_IMAGES" ]; then
  echo "No untagged images found."
  exit 0
fi

echo "Found untagged images:"
echo "$UNTAGGED_IMAGES"

echo "$UNTAGGED_IMAGES" | while read -r image; do
  # Convert the metadata.name field into the full image path
  CLEANED_IMAGE=$(echo "$image" | sed 's|projects/mlops-448109/locations/europe-west4/repositories/mlops-artifact-registry/dockerImages/|europe-west4-docker.pkg.dev/mlops-448109/mlops-artifact-registry/|')

  echo "Attempting to delete untagged image: $CLEANED_IMAGE"

  # Attempt to delete the untagged image
  if ! gcloud artifacts docker images delete "$CLEANED_IMAGE" --quiet; then
    echo "Warning: Failed to delete image $CLEANED_IMAGE. Skipping..."
  fi
done
