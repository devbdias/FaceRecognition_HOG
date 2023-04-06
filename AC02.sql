--BRUNO HENRIQUE DE OLIVEIRA DIAS 2100215

USE master;
GO

DROP DATABASE IF EXISTS ac02_1;
GO

CREATE DATABASE ac02_1;
GO

USE ac02_1;
GO

CREATE SYMMETRIC KEY symkey001
WITH ALGORITHM = AES_256
ENCRYPTION BY PASSWORD = 'Ratazana alada de Konoha';
GO

CREATE TABLE TBL_CTRL_ACESSO (
    Login VARCHAR(50) NOT NULL,
    Senha VARCHAR(255) NOT NULL,
    password_hint VARBINARY(255) NOT NULL
);
GO

CREATE FUNCTION fn_encrypt(@text VARCHAR(255))
RETURNS VARCHAR(255)
AS BEGIN
    RETURN HASHBYTES('SHA1', @text);
END;
GO

CREATE FUNCTION fn_decrypt(@hash VARCHAR(255))
RETURNS VARCHAR(255)
AS BEGIN
    RETURN CAST(HASHBYTES('SHA1', @hash) AS VARCHAR(255));
END;
GO

CREATE FUNCTION fn_hash(@text VARCHAR(255))
RETURNS VARBINARY(255)
AS BEGIN
    DECLARE @salt VARCHAR(255) = '5824#5$%jkHJhu2)!()';
    RETURN ENCRYPTBYKEY(KEY_GUID('symkey001'), @text, 1, CONVERT(varbinary(32), @salt));
END;
GO

CREATE PROCEDURE sp_authenticate
    @login VARCHAR(50),
    @senha VARCHAR(255)
AS BEGIN
    DECLARE @encrypted_password VARCHAR(255) = dbo.fn_encrypt(@senha);
    
    IF EXISTS (
        SELECT 1
        FROM TBL_CTRL_ACESSO
        WHERE Login = @login
        AND Senha = @encrypted_password
    )
    BEGIN
        SELECT 1;
    END
    ELSE
    BEGIN
        SELECT 0;
    END
END;
GO

CREATE PROCEDURE sp_get_password_hint
    @login VARCHAR(50)
AS BEGIN
    DECLARE @encrypted_hint VARBINARY(255);
    DECLARE @decrypted_hint VARCHAR(255);
    
    SELECT @encrypted_hint = password_hint
    FROM TBL_CTRL_ACESSO
    WHERE Login = @login;
    
    OPEN SYMMETRIC KEY symkey001
    DECRYPTION BY CERTIFICATE MyCertificate
    WITH PASSWORD = 'Ratazana alada de Konoha';
    
    SET @decrypted_hint = CONVERT(VARCHAR(255), DECRYPTBYKEY(@encrypted_hint));
    
    SELECT @decrypted_hint;
    
    CLOSE SYMMETRIC KEY symkey001;
END;
GO