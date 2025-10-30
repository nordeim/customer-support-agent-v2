"""
Session data encryption utilities.
Provides secure encryption/decryption for sensitive session data.

Version: 1.0.0
"""
import logging
import base64
from typing import Optional, Union
from datetime import datetime, timedelta

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Raised when encryption/decryption fails."""
    pass


class SessionEncryption:
    """
    Session data encryption using Fernet (symmetric encryption).
    
    Features:
    - AES-128 encryption in CBC mode
    - HMAC for integrity verification
    - Automatic key rotation support
    - Secure key derivation from passwords
    """
    
    def __init__(self, encryption_key: Optional[Union[str, bytes]] = None):
        """
        Initialize encryption with key.
        
        Args:
            encryption_key: Base64-encoded Fernet key or password
        """
        self.cipher: Optional[Fernet] = None
        self.key: Optional[bytes] = None
        
        if encryption_key:
            self._initialize_cipher(encryption_key)
    
    def _initialize_cipher(self, encryption_key: Union[str, bytes]) -> None:
        """
        Initialize Fernet cipher with key.
        
        Args:
            encryption_key: Encryption key (base64 or password)
        """
        try:
            # Convert string to bytes
            if isinstance(encryption_key, str):
                key_bytes = encryption_key.encode()
            else:
                key_bytes = encryption_key
            
            # Try to use as Fernet key directly
            try:
                self.cipher = Fernet(key_bytes)
                self.key = key_bytes
                logger.debug("Encryption cipher initialized with provided key")
            except Exception:
                # If not valid Fernet key, derive from password
                logger.debug("Deriving encryption key from password")
                self.key = self._derive_key_from_password(key_bytes)
                self.cipher = Fernet(self.key)
                
        except Exception as e:
            logger.error(f"Failed to initialize encryption cipher: {e}")
            raise EncryptionError(f"Cipher initialization failed: {e}")
    
    def _derive_key_from_password(
        self,
        password: bytes,
        salt: Optional[bytes] = None
    ) -> bytes:
        """
        Derive Fernet key from password using PBKDF2.
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation (generated if None)
            
        Returns:
            Base64-encoded Fernet key
        """
        if salt is None:
            # Use fixed salt for deterministic key derivation
            # In production, consider storing salt separately
            salt = b"customer_support_ai_salt_v1"
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    @staticmethod
    def generate_key() -> str:
        """
        Generate a new Fernet encryption key.
        
        Returns:
            Base64-encoded key as string
        """
        key = Fernet.generate_key()
        return key.decode()
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
            
        Raises:
            EncryptionError: If encryption fails
        """
        if not self.cipher:
            raise EncryptionError("Encryption cipher not initialized")
        
        try:
            # Convert string to bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Encrypt
            encrypted = self.cipher.encrypt(data)
            
            logger.debug(f"Encrypted {len(data)} bytes -> {len(encrypted)} bytes")
            return encrypted
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise EncryptionError(f"Encryption failed: {e}")
    
    def decrypt(self, encrypted_data: Union[str, bytes]) -> bytes:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
            
        Raises:
            EncryptionError: If decryption fails
        """
        if not self.cipher:
            raise EncryptionError("Encryption cipher not initialized")
        
        try:
            # Convert string to bytes if needed
            if isinstance(encrypted_data, str):
                encrypted_data = encrypted_data.encode('utf-8')
            
            # Decrypt
            decrypted = self.cipher.decrypt(encrypted_data)
            
            logger.debug(f"Decrypted {len(encrypted_data)} bytes -> {len(decrypted)} bytes")
            return decrypted
            
        except InvalidToken:
            logger.error("Decryption failed: Invalid token (wrong key or corrupted data)")
            raise EncryptionError("Decryption failed: Invalid token")
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise EncryptionError(f"Decryption failed: {e}")
    
    def encrypt_string(self, data: str) -> str:
        """
        Encrypt string and return base64-encoded result.
        
        Args:
            data: String to encrypt
            
        Returns:
            Base64-encoded encrypted string
        """
        encrypted = self.encrypt(data)
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_string(self, encrypted_data: str) -> str:
        """
        Decrypt base64-encoded encrypted string.
        
        Args:
            encrypted_data: Base64-encoded encrypted string
            
        Returns:
            Decrypted string
        """
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        decrypted = self.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')
    
    def rotate_key(self, new_key: Union[str, bytes]) -> None:
        """
        Rotate encryption key.
        
        Args:
            new_key: New encryption key
            
        Note:
            Existing encrypted data will need to be re-encrypted with new key
        """
        logger.info("Rotating encryption key")
        self._initialize_cipher(new_key)
    
    def verify_key(self, test_data: str = "test") -> bool:
        """
        Verify encryption key by performing encrypt/decrypt roundtrip.
        
        Args:
            test_data: Test data to use
            
        Returns:
            True if key is valid
        """
        try:
            encrypted = self.encrypt(test_data)
            decrypted = self.decrypt(encrypted)
            return decrypted.decode('utf-8') == test_data
        except Exception as e:
            logger.error(f"Key verification failed: {e}")
            return False


class TimestampedEncryption(SessionEncryption):
    """
    Encryption with built-in timestamp validation.
    Prevents replay attacks by validating encryption age.
    """
    
    def __init__(
        self,
        encryption_key: Optional[Union[str, bytes]] = None,
        max_age_seconds: int = 3600
    ):
        """
        Initialize timestamped encryption.
        
        Args:
            encryption_key: Encryption key
            max_age_seconds: Maximum age for encrypted data
        """
        super().__init__(encryption_key)
        self.max_age_seconds = max_age_seconds
    
    def encrypt_with_timestamp(self, data: Union[str, bytes]) -> bytes:
        """
        Encrypt data with timestamp.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data with embedded timestamp
        """
        if not self.cipher:
            raise EncryptionError("Encryption cipher not initialized")
        
        try:
            # Convert to bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Encrypt with TTL
            encrypted = self.cipher.encrypt_at_time(
                data,
                current_time=int(datetime.utcnow().timestamp())
            )
            
            return encrypted
            
        except Exception as e:
            logger.error(f"Timestamped encryption failed: {e}")
            raise EncryptionError(f"Timestamped encryption failed: {e}")
    
    def decrypt_with_timestamp(self, encrypted_data: Union[str, bytes]) -> bytes:
        """
        Decrypt data and validate timestamp.
        
        Args:
            encrypted_data: Encrypted data with timestamp
            
        Returns:
            Decrypted data
            
        Raises:
            EncryptionError: If data is too old or decryption fails
        """
        if not self.cipher:
            raise EncryptionError("Encryption cipher not initialized")
        
        try:
            # Convert to bytes
            if isinstance(encrypted_data, str):
                encrypted_data = encrypted_data.encode('utf-8')
            
            # Decrypt with TTL validation
            decrypted = self.cipher.decrypt_at_time(
                encrypted_data,
                ttl=self.max_age_seconds,
                current_time=int(datetime.utcnow().timestamp())
            )
            
            return decrypted
            
        except InvalidToken as e:
            if "too old" in str(e).lower():
                logger.warning(f"Encrypted data expired (max_age={self.max_age_seconds}s)")
                raise EncryptionError("Encrypted data has expired")
            else:
                logger.error(f"Invalid token: {e}")
                raise EncryptionError("Invalid encrypted data")
        except Exception as e:
            logger.error(f"Timestamped decryption failed: {e}")
            raise EncryptionError(f"Timestamped decryption failed: {e}")


def create_encryption_instance(
    encryption_key: Optional[Union[str, bytes]] = None,
    use_timestamp: bool = False,
    max_age_seconds: int = 3600
) -> Union[SessionEncryption, TimestampedEncryption]:
    """
    Factory function to create encryption instance.
    
    Args:
        encryption_key: Encryption key
        use_timestamp: Whether to use timestamped encryption
        max_age_seconds: Maximum age for timestamped encryption
        
    Returns:
        Encryption instance
    """
    if use_timestamp:
        return TimestampedEncryption(encryption_key, max_age_seconds)
    else:
        return SessionEncryption(encryption_key)


__all__ = [
    'SessionEncryption',
    'TimestampedEncryption',
    'EncryptionError',
    'create_encryption_instance'
]
