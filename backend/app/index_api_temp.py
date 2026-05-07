@router.delete("/documents/delete")
def delete_document(filename: str, retry: bool = True):
    """Delete a specific document by its filename from the vector store."""
    safe_name = os.path.basename(filename)
    client = get_qdrant_client()
    try:
        # Check if collection exists
        try:
            collections = client.get_collections().collections
            if not any(c.name == QDRANT_COLLECTION for c in collections):
                return {"status": "success", "message": f"Document '{safe_name}' not found (index empty)"}
        except Exception as e:
            logger.warning(f"Collection check failed: {e}")

        # Ensure payload index
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="metadata.source_file",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

        # Perform deletion with explicit filter
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="metadata.source_file",
                            match=models.MatchValue(value=safe_name),
                        ),
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=safe_name),
                        ),
                    ]
                )
            ),
        )
        
        # Physical file cleanup
        file_path = os.path.join(UPLOAD_DIR, safe_name)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Physical file delete failed: {e}")
            
    except Exception as exc:
        if retry and is_qdrant_client_closed_error(exc):
            logger.warning("Qdrant client closed; recreating for delete retry")
            from .qdrant_conn import recreate_qdrant_client
            recreate_qdrant_client()
            return delete_document(filename, retry=False)
            
        logger.error(f"Failed to delete document '{safe_name}': {type(exc).__name__}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Delete failed: {type(exc).__name__}: {str(exc)}",
        )
    
    return {"status": "success", "message": f"Document '{safe_name}' deleted"}
