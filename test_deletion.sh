gcloud artifacts docker images list \
    europe-west4-docker.pkg.dev/mlops-448109/mlops-artifact-registry \
    --format="json" | jq -r '.[] | select(.tags == "") | .metadata.name' | while read -r image; do

    # Extract the repository and digest
    CLEANED_IMAGE=$(echo "$image" | sed 's|projects/mlops-448109/locations/europe-west4/repositories/mlops-artifact-registry/dockerImages/|europe-west4-docker.pkg.dev/mlops-448109/mlops-artifact-registry/|')

    echo "Deleting untagged image: $CLEANED_IMAGE"

    # Delete the untagged image without affecting other tags
    gcloud artifacts docker images delete "$CLEANED_IMAGE" --quiet
done
